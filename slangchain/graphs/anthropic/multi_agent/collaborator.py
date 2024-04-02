"""Collaboration class"""
from typing import Dict, Sequence, List, Optional, Any
import logging
import functools
import json

from langchain_core.runnables.base import RunnableSequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain.chains.base import Chain
from langchain.tools.base import BaseTool

from langgraph.pregel import Pregel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from slangchain.graphs.anthropic.schemas import (
  CollaboratorAgentState as AgentState,
  CollaboratorNodeTool
)
from slangchain.llms.chat_anthropic_functions import ChatAnthropicFunctions
from slangchain.tools.render import format_tool_to_anthropic_function

TOOLS_PROMPT = (
  "You are a helpful AI assistant, collaborating with other assistants."
  " Use the provided tools to progress towards answering the question."
  " If you are unable to fully answer, that's OK, another assistant with different tools "
  " will help where you left off. Execute what you can to make progress."
  " If you or any of the other assistants have the final answer or deliverable,"
  " prefix your response with FINAL ANSWER so the team knows to stop."
  " You have access to the following tools: {tool_names}.\n{system_message}"
)

CALL_TOOL_NODE_NAME = "call_tool"

logger = logging.getLogger(__name__)


class AgentNode():
  """AgentNode"""
  def agent_node(
      self,
      state: AgentState,
      agent: RunnableSequence,
      name: str) -> AgentState:
    """agent_node"""
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state

    logger.debug("\n================ agent_node sender: %s ================", name)
    logger.debug("result: %s | %s", result.type, result)
    if isinstance(result, AIMessage) and result.additional_kwargs:
      pass
    else:
      result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    logger.debug("changed result: %s | %s", result.type, result)

    logger.debug("================================\n")

    return {
      "messages": [result],
      # Since we have a strict workflow, we can
      # track the sender so we know who to pass to next.
      "sender": name,
    }


class ToolNode():
  """ToolNode"""

  def __init__(
    self,
    tool_executor: ToolExecutor):
    self.tool_executor = tool_executor

  def tool_node(self, state: AgentState):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    logger.debug("================ _tool_node sender: %s ================", state["sender"])
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    logger.debug("tool_node_last_message: %s | %s\n", last_message.type, last_message)
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
      tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = self.tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = AIMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    logger.debug("tool_node_function_message: %s | %s\n", function_message.type, function_message)
    logger.debug("================================")
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


class Router():
  """Router"""
  def router(
    self,
    state: AgentState):
    """router"""
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    logger.debug("router last_message: %s | %s", last_message.type, last_message)
    if "function_call" in last_message.additional_kwargs:
      # The previus agent is invoking a tool
      logger.debug("router: CALL_TOOL")
      return "call_tool"
    if "FINAL ANSWER" in last_message.content:
      # Any agent decided the work is done
      logger.debug("router: END")
      return "end"
    logger.debug("router: CONTINUE")
    return "continue"


class Collaborator(Chain):
  """Collaboration"""
  llm: ChatAnthropicFunctions
  node_tools: List[CollaboratorNodeTool]
  workflow: Optional[StateGraph] = Field(default = None)
  graph: Optional[Pregel] = Field(default = None)
  recursion_limit: Optional[int] = Field(default = 100)

  input_key: str = "input"  #: :meta private:
  output_key: str = "output"  #: :meta private:

  class Config:
    """BaseModel config"""
    extra = Extra.allow
    arbitrary_types_allowed = True


  @root_validator()
  def validate_class_objects(cls, values: Dict) -> Dict:
    """Validate that chains are all single input/output."""
    if not isinstance(values["llm"], ChatAnthropicFunctions):
      raise TypeError("llm must be of instance ChatAnthropicFunctions")
    if not values["node_tools"]:
      return ValueError("tools list empty")
    if not all(isinstance(tool, CollaboratorNodeTool) for tool in values["node_tools"]):
      raise TypeError("tools all must be of CollaboratorNodeTool type")
    return values


  @property
  def input_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return [self.input_key]


  @property
  def output_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return [self.output_key]


  def _add_workflow_nodes(
    self,
    llm: ChatAnthropicFunctions,
    workflow: StateGraph,
    node_tools: List[CollaboratorNodeTool]) -> StateGraph:
    """add tools to workflow"""

    agent_node = AgentNode()
    for node_tool in node_tools:
      tool_agent = self._create_tools_agent(
        llm = llm,
        tools = [node_tool.tool],
        system_prompt = node_tool.description
      )

      tool_node = functools.partial(
        agent_node.agent_node,
        agent = tool_agent,
        name = node_tool.name)
      workflow.add_node(node_tool.name, tool_node)

    tools = [node_tool.tool for node_tool in node_tools]
    tool_executor = ToolExecutor(tools)

    call_tool_node = ToolNode(tool_executor = tool_executor)
    workflow.add_node(CALL_TOOL_NODE_NAME, call_tool_node.tool_node)

    return workflow


  def _add_workflow_edges(
    self,
    workflow: StateGraph,
    node_tools: List[CollaboratorNodeTool]
  ) -> StateGraph:
    """add workflow edges"""
    node_tools_dict = {}
    router = Router()

    for node_tool in node_tools:
      if not node_tool.conditional_edge_node:
        continue
      workflow.add_conditional_edges(
        node_tool.name,
        router.router,
        {
          "continue": node_tool.conditional_edge_node,
          CALL_TOOL_NODE_NAME: CALL_TOOL_NODE_NAME,
          "end": END
        }
      )
      node_tools_dict[node_tool.name] = node_tool.name

    workflow.add_conditional_edges(
      CALL_TOOL_NODE_NAME,
      lambda x: x["sender"],
      node_tools_dict)

    return workflow


  def _set_entry_point(
    self,
    workflow: StateGraph,
    node_tools: List[CollaboratorNodeTool]
  ) -> StateGraph:
    """add workflow edges"""
    entry_point = node_tools[0].name
    for node_tool in node_tools:
      if node_tool.entrypoint_flag:
        entry_point = node_tool.name
        break
    workflow.set_entry_point(entry_point)
    return workflow


  def _create_tools_agent(
    self,
    llm: ChatAnthropicFunctions,
    tools: List[BaseTool],
    system_prompt: str
  ) -> RunnableSequence:
    """create tools agent"""

    functions = [format_tool_to_anthropic_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages([
      ("system", TOOLS_PROMPT,),
      MessagesPlaceholder(variable_name = "messages"),])

    prompt = prompt.partial(system_message = system_prompt)
    prompt = prompt.partial(tool_names = ", ".join([tool.name for tool in tools]))
    return prompt | llm.bind(functions = functions)


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    self.workflow = StateGraph(AgentState)
    self.workflow = self._add_workflow_nodes(
      llm = self.llm,
      workflow = self.workflow,
      node_tools = self.node_tools,
    )
    self.workflow = self._add_workflow_edges(
      workflow = self.workflow,
      node_tools = self.node_tools,
    )
    self.workflow = self._set_entry_point(
      workflow = self.workflow,
      node_tools = self.node_tools,
    )
    return self.workflow

  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:
    message = inputs[self.input_key]

    self.workflow = self.init_workflow_nodes()
    self.graph = self.workflow.compile()

    result : Dict[str, Any] = {}

    for graph_stream in self.graph.stream(
      {
        "messages": [
            HumanMessage(content=message)
        ]
      },
      {"recursion_limit": self.recursion_limit},
    ):
      if "__end__" not in graph_stream:
        logger.info(graph_stream)
        logger.info("----")
      result = graph_stream.get("__end__", {})

    return {self.output_key: result}


  @classmethod
  def from_llm_and_node_tools(
    cls,
    llm: ChatAnthropicFunctions,
    node_tools: Sequence[CollaboratorNodeTool],
    recursion_limit: Optional[int] = 100,
  ) -> "Collaborator":
    """Construct a Collaborator from an LLM and tools."""

    return cls(
      llm = llm,
      recursion_limit = recursion_limit,
      node_tools = node_tools,
    )
