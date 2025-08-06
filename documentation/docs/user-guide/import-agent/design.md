# Import Agent Design

Design overview for the import-agent utility, explaining the choices behind the generated agent.

## Feature Support

| Bedrock Agent Feature                               | Langchain | Strands | AgentCore               |
|-----------------------------------------------------|-------------------------------------|----------------------------------|--------------------------------------------------|
| *Guardrails*                                        | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Orchestration (via reAct)*                         | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Knowledge Bases*                                   | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Code Interpreter*                                  | SUPPORTED                            | SUPPORTED                         | SUPPORTED: 1P Code Interpreter                    |
| *Lambda Function Definitions*                       | SUPPORTED                            | SUPPORTED                         | SUPPORTED: AgentCore Gateway                  |
| *Lambda OpenAPI Definitions*                        | SUPPORTED                            | SUPPORTED                         | SUPPORTED: AgentCore Gateway                  |
| *Return of Control*                                 | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Short Term (Conversational) Memory*                | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Long Term (Cross-Session) Memory*                  | SUPPORTED                            | SUPPORTED                         | SUPPORTED: AgentCore Memory                      |
| *Session Summarization*                             | SUPPORTED                            | SUPPORTED                         | SUPPORTED: AgentCore Memory                      |
| *Pre Processing Step*                               | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Post Processing Step*                              | SUPPORTED                            | SUPPORTED                         |                                                  |
| *KB Generation Routing/Optimizations*               | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Idle Timeouts*                                     | SUPPORTED                            | SUPPORTED                         |                                                  |
| *User Input (as a tool)*                            | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Traces*                                            | SUPPORTED                            | SUPPORTED                         | SUPPORTED: AgentCore Observability               |
| *Multi-Agent Collaboration - Supervisor Mode*       | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Multi-Agent Collaboration - Routing Mode*          | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Multi-Agent Collaboration - Conversation Relay*    | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Custom Bedrock Model Usage*                        | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Chat Interface (via CLI)*                          | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Custom Inference Configurations*                   | SUPPORTED                            | SUPPORTED                         |                                                  |
| *Agent Deployment*                                  | N/A                                  | N/A                              | SUPPORTED: AgentCore Runtime                     |
| *Lambda Parsing and Orchestration*                  | N/A                                  | N/A                              |                                                  |


## Action Groups -> AgentCore Gateway Target

In Bedrock Agents, users can define Action Groups for their agent. An Action Group is a collection of tools either executed via Lambda or with a local callback (Return of Control) and defined with an OpenAPI spec or structured function schema. BR Agents, at runtime, will call your Lambda function with an event of the following format, depending on which schema option you chose: [https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html).

In AgentCore Gateway, we create one gateway for the output agent. Each action group is equivalent to one target in this gateway, and each function or path/method is one tool in this target. In order to maintain backwards compatibility with the action group lambda functions, we create a proxy Lambda function that serves as the Lambda for all the gateway's targets. This proxy handles an incoming tool call in Gateway's format, finds the correct end action group Lambda function to call for that tool, and then reformats data to call that Lambda function and return its output.

If AgentCore Gateway is opted out, then the utility generates local tools for each function and path/method, exposes their relevant argument schema to the agent via Pydantic model generation, then has each tool format its input correctly to call the relevant action group lambda. This also applies for ROC action groups, where such a tool invocation will ask the user for local input to continue execution.

## Session Memory -> AgentCore Memory

In Bedrock Agents, users can enable Long Term Memory for their agent. This is based on session summarization, where BR Agents uses an LLM to summarize the conversation’s discussion topics based on a session’s messages. A max number of sessions and max days threshold can be set. In the orchestration prompt in BR agents, a synopsis of the long-term memory, consisting of multiple of these session summaries, is injected.

In AgentCore Memory, we create a memory store adopting the summarization strategy:

```python
{
    "summaryMemoryStrategy": {
        "name": "SessionSummarizer",
        "namespaces": ["/summaries/{actorId}/{sessionId}"],
    }
}
```

On each entrypoint invocation, we save formatted messages to this memory store by creating an event with the correct user id and session id. During agent initialization within the agent update loop (`get_agent`), we retrieve the top session summary memories from the memory store, format the memories correctly, and inject them as a memory synopsis into the agent's system prompt.

If AgentCore Memory is opted out, then we replicate the behavior with a local long term memory manager, which uses a memory summarization LLM and the memory summarization BR agents prompt to create and manage session summaries. The generated summaries are saved and maintained in a local session summaries JSON file.

## Code Interpreter -> AgentCore Code Interpreter

In Bedrock Agents, users can enable Code Interpreter. This gives a Bedrock Agent access to a sandbox for it to write, troubleshoot, and return the output of code.

We use AgentCore Code Interpreter for an equivalent experience. The utility defines a `code_tool` which creates a code sandbox session and defines a sub-agent with access to code interpreter operations (as tools). This sub-agent is fed the code tool's input query and runs in a loop using the sandbox operations to accomplish the coding task and return the output.

When AgentCore Code Interpreter is opted-out, we use open-interpreter, an open source and local code interpreter that can write, execute, and troubleshoot code.

## Traces -> AgentCore Observability

In Bedrock Agents, users can view traces that describe pre/post processing steps, routing classifier steps, guardrail invocation, agent orchestration, and other information. These traces are in a format specific to Bedrock Agents, and can be viewed either in the console or as output of an invoke_agent call to a BR agent.

The equivalent for this with AgentCore is to use AgentCore Observability (if not opted-out). For both Langchain and Strands, the agent will output OTEL logs on a session, trace, and span level. These logs are captured by AgentCore Observability when the agent is deployed to AgentCore Runtime, and the logs will be visible in CloudWatch under the GenAI Observability section.

## Guardrails

In Bedrock Agents, users can add Bedrock Guardrails to their agent. This applies the guardrail on the model level and Bedrock Agents will have defined behavior for when a guardrail is invoked to redact or block an input. Equivalently, the utility applies the same guardrail on Bedrock models, as there is support for this in both Langchain and Strands.

## Knowledge Bases

In Bedrock Agents, users can add existing Knowledge Bases (defined via Bedrock Knowledge Bases) to their agent via the Console or SDK. The AgentCore + Langchain/Strands equivalent of this feature is to use each of those KBs to define a KB retrieval tool for an agent to use. This tool uses the AWS SDK to retrieve from connected knowledge bases using a query decided by the agent and then returns the document results for the agent to use.

## Multi-Agent Collaboration

In Bedrock Agents, users can promote an agent and add collaborators to it. This hierarchy can be up to 5 levels deep. A collaborator can receive shared conversation history from the parent, and can be invoked with routing mode (parent uses a routing classifier prompt to find a relevant collaborator for a user query) or supervisor mode (an agents-as-tools approach).

The utility's approach to this is to recursively translate a parent agent and its children, and then orchestrate them together via an Agents-as-Tools approach by default. If conversation sharing is enabled, then the parent will inject its state into the child's via these collaboration tools. If routing mode is enabled for the parent, then the parent agent uses a routing classifier prompt, before orchestration, to invoke a relevant child agent. In AgentCore Runtime, the code for a parent agent and its children are packaged together, in the same container image, to enable this setup.

## Code Structure

In general, the generated agents follow the structure below:

1. Imports
2. Model configurations
3. Prompt definitions
4. Action Group / MCP Tools
5. Knowledge Base Tools
6. Agent Initialization and Update Loop
7. Agent Invocation Logic
8. AgentCore Runtime Entrypoint
9. CLI for Local Agent Testing

## Prompt Generation, Testing, and Agent Similarity

Bedrock Agents prompts are constructed at runtime, by substituting in template fixtures (ie. variables in the prompt). For example, an orchestration prompt may have the fixture called knowledge_base_guidelines. This variable is filled in depending on the model provider and model version in use.

To approximate the same behavior and deliver translated agents that are functionally equivalent, the utility uses ONE collection of template fixtures (`template_fixtures_merged.json`) and substitutes them in to build the correct prompts.

This, along with all the design choices above, yield translated agents that are very similar semantically and functionally to the original Bedrock Agents. Below are functional testing results on a sample of 400 randomly generated and translated BR agents on a shared prompt suite:

- ~95% match in average semantic similarity of responses via text embedding
- 97.50% match in Langchain agent usage of KBs, collaborators, guardrails, action groups
- 100% match in Strands agent usage of KBs, collaborators, guardrails, action groups
