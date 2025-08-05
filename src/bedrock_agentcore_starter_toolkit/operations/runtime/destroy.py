"""Destroy operation - removes Bedrock AgentCore resources from AWS."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from ...services.runtime import BedrockAgentCoreClient
from ...utils.runtime.config import load_config, save_config
from ...utils.runtime.schema import BedrockAgentCoreAgentSchema, BedrockAgentCoreConfigSchema
from .models import DestroyResult

log = logging.getLogger(__name__)


def destroy_bedrock_agentcore(
    config_path: Path,
    agent_name: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> DestroyResult:
    """Destroy Bedrock AgentCore resources.

    Args:
        config_path: Path to the configuration file
        agent_name: Name of the agent to destroy (default: use default agent)
        dry_run: If True, only show what would be destroyed without actually doing it
        force: If True, skip confirmation prompts

    Returns:
        DestroyResult: Details of what was destroyed or would be destroyed

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If agent is not found or not deployed
        RuntimeError: If destruction fails
    """
    log.info("Starting destroy operation for agent: %s (dry_run=%s)", agent_name or "default", dry_run)

    try:
        # Load configuration
        project_config = load_config(config_path)
        agent_config = project_config.get_agent_config(agent_name)
        
        if not agent_config:
            raise ValueError(f"Agent '{agent_name or 'default'}' not found in configuration")

        # Initialize result
        result = DestroyResult(agent_name=agent_config.name, dry_run=dry_run)

        # Check if agent is deployed
        if not agent_config.bedrock_agentcore:
            result.warnings.append("Agent is not deployed, nothing to destroy")
            return result

        # Initialize AWS session and clients
        session = boto3.Session(region_name=agent_config.aws.region)
        
        # 1. Destroy Bedrock AgentCore endpoint (if exists)
        _destroy_agentcore_endpoint(session, agent_config, result, dry_run)
        
        # 2. Destroy Bedrock AgentCore agent
        _destroy_agentcore_agent(session, agent_config, result, dry_run)
        
        # 3. Remove ECR images (specific tags only)
        _destroy_ecr_images(session, agent_config, result, dry_run)
        
        # 4. Remove CodeBuild project
        _destroy_codebuild_project(session, agent_config, result, dry_run)
        
        # 5. Remove IAM execution role (if not used by other agents)
        _destroy_iam_role(session, project_config, agent_config, result, dry_run)
        
        # 6. Clean up configuration
        if not dry_run and not result.errors:
            _cleanup_agent_config(config_path, project_config, agent_config.name, result)

        log.info("Destroy operation completed. Resources removed: %d, Warnings: %d, Errors: %d",
                len(result.resources_removed), len(result.warnings), len(result.errors))
        
        return result

    except Exception as e:
        log.error("Destroy operation failed: %s", str(e))
        raise RuntimeError(f"Destroy operation failed: {e}") from e


def _destroy_agentcore_endpoint(
    session: boto3.Session,
    agent_config: BedrockAgentCoreAgentSchema,
    result: DestroyResult,
    dry_run: bool,
) -> None:
    """Destroy Bedrock AgentCore endpoint."""
    if not agent_config.bedrock_agentcore:
        return

    try:
        client = BedrockAgentCoreClient(session)
        
        # Get agent details to find endpoint
        agent_id = agent_config.bedrock_agentcore.agent_id
        if not agent_id:
            result.warnings.append("No agent ID found, skipping endpoint destruction")
            return

        if dry_run:
            result.resources_removed.append(f"AgentCore endpoint for agent {agent_id} (DRY RUN)")
            return

        # Try to get and delete endpoint
        try:
            endpoints = client.list_agent_runtime_endpoints(agent_runtime_arn=agent_config.bedrock_agentcore.agent_arn)
            for endpoint in endpoints.get("agentRuntimeEndpointSummaries", []):
                endpoint_arn = endpoint.get("agentRuntimeEndpointArn")
                if endpoint_arn:
                    client.delete_agent_runtime_endpoint(agent_runtime_endpoint_arn=endpoint_arn)
                    result.resources_removed.append(f"AgentCore endpoint: {endpoint_arn}")
                    log.info("Deleted AgentCore endpoint: %s", endpoint_arn)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ["ResourceNotFoundException", "NotFound"]:
                result.warnings.append(f"Failed to delete endpoint: {e}")
                log.warning("Failed to delete endpoint: %s", e)

    except Exception as e:
        result.warnings.append(f"Error during endpoint destruction: {e}")
        log.warning("Error during endpoint destruction: %s", e)


def _destroy_agentcore_agent(
    session: boto3.Session,
    agent_config: BedrockAgentCoreAgentSchema,
    result: DestroyResult,
    dry_run: bool,
) -> None:
    """Destroy Bedrock AgentCore agent."""
    if not agent_config.bedrock_agentcore or not agent_config.bedrock_agentcore.agent_arn:
        result.warnings.append("No agent ARN found, skipping agent destruction")
        return

    try:
        client = BedrockAgentCoreClient(session)
        agent_arn = agent_config.bedrock_agentcore.agent_arn

        if dry_run:
            result.resources_removed.append(f"AgentCore agent: {agent_arn} (DRY RUN)")
            return

        # Delete the agent
        try:
            client.delete_agent_runtime(agent_runtime_arn=agent_arn)
            result.resources_removed.append(f"AgentCore agent: {agent_arn}")
            log.info("Deleted AgentCore agent: %s", agent_arn)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ["ResourceNotFoundException", "NotFound"]:
                result.errors.append(f"Failed to delete agent {agent_arn}: {e}")
                log.error("Failed to delete agent: %s", e)
            else:
                result.warnings.append(f"Agent {agent_arn} not found (may have been deleted already)")

    except Exception as e:
        result.errors.append(f"Error during agent destruction: {e}")
        log.error("Error during agent destruction: %s", e)


def _destroy_ecr_images(
    session: boto3.Session,
    agent_config: BedrockAgentCoreAgentSchema,
    result: DestroyResult,
    dry_run: bool,
) -> None:
    """Remove ECR images for this specific agent."""
    if not agent_config.aws.ecr_repository:
        result.warnings.append("No ECR repository configured, skipping image cleanup")
        return

    try:
        ecr_client = session.client("ecr")
        ecr_uri = agent_config.aws.ecr_repository
        
        # Extract repository name from URI
        # Format: account.dkr.ecr.region.amazonaws.com/repo-name
        repo_name = ecr_uri.split("/")[-1]

        if dry_run:
            result.resources_removed.append(f"ECR images in repository: {repo_name} (DRY RUN)")
            return

        try:
            # List images with latest tag for this agent
            response = ecr_client.list_images(
                repositoryName=repo_name,
                filter={"tagStatus": "TAGGED"}
            )
            
            images_to_delete = []
            for image in response.get("imageDetails", []):
                # Only delete images tagged as 'latest' or with the agent name
                tags = image.get("imageTags", [])
                if "latest" in tags or agent_config.name in tags:
                    images_to_delete.append({"imageTag": "latest"})
                    break  # Only delete latest for safety

            if images_to_delete:
                ecr_client.batch_delete_image(
                    repositoryName=repo_name,
                    imageIds=images_to_delete
                )
                result.resources_removed.append(f"ECR images: {len(images_to_delete)} images from {repo_name}")
                log.info("Deleted %d ECR images from %s", len(images_to_delete), repo_name)
            else:
                result.warnings.append(f"No ECR images found to delete in {repo_name}")

        except ClientError as e:
            if e.response["Error"]["Code"] not in ["RepositoryNotFoundException"]:
                result.warnings.append(f"Failed to delete ECR images: {e}")
                log.warning("Failed to delete ECR images: %s", e)
            else:
                result.warnings.append(f"ECR repository {repo_name} not found")

    except Exception as e:
        result.warnings.append(f"Error during ECR cleanup: {e}")
        log.warning("Error during ECR cleanup: %s", e)


def _destroy_codebuild_project(
    session: boto3.Session,
    agent_config: BedrockAgentCoreAgentSchema,
    result: DestroyResult,
    dry_run: bool,
) -> None:
    """Remove CodeBuild project for this agent."""
    try:
        codebuild_client = session.client("codebuild")
        project_name = f"bedrock-agentcore-{agent_config.name}-builder"

        if dry_run:
            result.resources_removed.append(f"CodeBuild project: {project_name} (DRY RUN)")
            return

        try:
            codebuild_client.delete_project(name=project_name)
            result.resources_removed.append(f"CodeBuild project: {project_name}")
            log.info("Deleted CodeBuild project: %s", project_name)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ["ResourceNotFoundException"]:
                result.warnings.append(f"Failed to delete CodeBuild project {project_name}: {e}")
                log.warning("Failed to delete CodeBuild project: %s", e)
            else:
                result.warnings.append(f"CodeBuild project {project_name} not found")

    except Exception as e:
        result.warnings.append(f"Error during CodeBuild cleanup: {e}")
        log.warning("Error during CodeBuild cleanup: %s", e)


def _destroy_iam_role(
    session: boto3.Session,
    project_config: BedrockAgentCoreConfigSchema,
    agent_config: BedrockAgentCoreAgentSchema,
    result: DestroyResult,
    dry_run: bool,
) -> None:
    """Remove IAM execution role only if not used by other agents."""
    if not agent_config.aws.execution_role:
        result.warnings.append("No execution role configured, skipping IAM cleanup")
        return

    try:
        iam_client = session.client("iam")
        role_arn = agent_config.aws.execution_role
        role_name = role_arn.split("/")[-1]

        # Check if other agents use the same role
        other_agents_using_role = [
            name for name, agent in project_config.agents.items()
            if name != agent_config.name and agent.aws.execution_role == role_arn
        ]

        if other_agents_using_role:
            result.warnings.append(
                f"IAM role {role_name} is used by other agents: {other_agents_using_role}. Not deleting."
            )
            return

        if dry_run:
            result.resources_removed.append(f"IAM execution role: {role_name} (DRY RUN)")
            return

        try:
            # Delete attached policies first
            try:
                policies = iam_client.list_attached_role_policies(RoleName=role_name)
                for policy in policies.get("AttachedPolicies", []):
                    iam_client.detach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy["PolicyArn"]
                    )
            except ClientError:
                pass  # Continue if policy detachment fails

            # Delete inline policies
            try:
                inline_policies = iam_client.list_role_policies(RoleName=role_name)
                for policy_name in inline_policies.get("PolicyNames", []):
                    iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            except ClientError:
                pass  # Continue if inline policy deletion fails

            # Delete the role
            iam_client.delete_role(RoleName=role_name)
            result.resources_removed.append(f"IAM execution role: {role_name}")
            log.info("Deleted IAM role: %s", role_name)

        except ClientError as e:
            if e.response["Error"]["Code"] not in ["NoSuchEntity"]:
                result.warnings.append(f"Failed to delete IAM role {role_name}: {e}")
                log.warning("Failed to delete IAM role: %s", e)
            else:
                result.warnings.append(f"IAM role {role_name} not found")

    except Exception as e:
        result.warnings.append(f"Error during IAM cleanup: {e}")
        log.warning("Error during IAM cleanup: %s", e)


def _cleanup_agent_config(
    config_path: Path,
    project_config: BedrockAgentCoreConfigSchema,
    agent_name: str,
    result: DestroyResult,
) -> None:
    """Remove agent configuration from the config file."""
    try:
        # Clear the bedrock_agentcore deployment info but keep the agent config
        if agent_name in project_config.agents:
            agent_config = project_config.agents[agent_name]
            agent_config.bedrock_agentcore = None
            
            # Save updated configuration
            save_config(project_config, config_path)
            result.resources_removed.append(f"Agent deployment configuration: {agent_name}")
            log.info("Cleared deployment configuration for agent: %s", agent_name)
        else:
            result.warnings.append(f"Agent {agent_name} not found in configuration")

    except Exception as e:
        result.warnings.append(f"Failed to update configuration: {e}")
        log.warning("Failed to update configuration: %s", e)