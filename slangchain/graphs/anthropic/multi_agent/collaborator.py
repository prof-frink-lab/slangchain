"""Collaboration class"""
from typing import Dict, Sequence, List, Optional, Callable, Any
import logging
import functools

from langchain_core.runnables.base import RunnableSequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain.chains.base import Chain
from langchain.tools.base import BaseTool

from langgraph.pregel import Pregel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from slangchain.graphs.anthropic.schemas import (
  CollaboratorAgentState as AgentState,
  CollaboratorToolsNode
)
from slangchain.llms.chat_anthropic_tools import ChatAnthropicTools

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

    result: BaseMessage = None
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state

    logger.debug("\n================ agent_node sender: %s ================", name)
    logger.debug("result: %s | %s", result.type, result)
    tool_use_flag = False
    if isinstance(result.content, list):
      if result.content[-1].get("type") == "tool_use":
        tool_use_flag = True
    if isinstance(result, AIMessage) and tool_use_flag:
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
    tool_id = None
    tool_name = None
    tool_input = {}

    logger.debug("================ _tool_node sender: %s ================", state["sender"])
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    logger.debug("tool_node_last_message: %s | %s\n", last_message.type, last_message)
    # We construct an ToolInvocation from the the tool_use message
    if isinstance(last_message.content, list):
      if last_message.content[-1].get("type") == "tool_use":
        tool_input = last_message.content[-1].get("input")
        tool_name = last_message.content[-1].get("name")
        tool_id = last_message.content[-1].get("id")


    if not tool_name:
      return {"messages": []}

    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = self.tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    tool_message = ToolMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool, tool_call_id = tool_id
    )
    logger.debug("tool_node_function_message: %s | %s\n", tool_message.type, tool_message)
    logger.debug("================================")
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}


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
    if isinstance(last_message.content, list):
      if last_message.content[-1].get("type") == "tool_use":
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
  model_name: str
  tools_nodes: List[CollaboratorToolsNode]
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
    if "claude-3" not in values["model_name"]:
      raise TypeError("model_name must start with claude-3")
    if not values["tools_nodes"]:
      return ValueError("tools list empty")
    if not all(isinstance(tool, CollaboratorToolsNode) for tool in values["tools_nodes"]):
      raise TypeError("tools all must be of CollaboratorToolsNode type")
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


  def _create_tools_node(
    self,
    llm: ChatAnthropicTools,
    agent_node: AgentNode,
    tools_node: CollaboratorToolsNode) -> Callable:
    """_create_tools_node"""
    tool_agent = self._create_tools_agent(
      llm = llm,
      tools = tools_node.tools,
      system_prompt = tools_node.prompt
    )

    tool_node = functools.partial(
      agent_node.agent_node,
      agent = tool_agent,
      name = tools_node.name)

    return tool_node


  def _add_workflow_nodes(
    self,
    model_name: str,
    workflow: StateGraph,
    tools_nodes: List[CollaboratorToolsNode]) -> StateGraph:
    """add tools to workflow"""

    agent_node = AgentNode()
    for tools_node in tools_nodes:
      llm = ChatAnthropicTools(model_name = model_name)
      workflow_node = self._create_tools_node(
        llm, agent_node, tools_node
      )
      workflow.add_node(tools_node.name, workflow_node)

    tools = [tool for tools_node in tools_nodes for tool in tools_node.tools]
    tool_executor = ToolExecutor(tools)

    call_tools_node = ToolNode(tool_executor = tool_executor)
    workflow.add_node(CALL_TOOL_NODE_NAME, call_tools_node.tool_node)

    return workflow


  def _add_workflow_edges(
    self,
    workflow: StateGraph,
    tools_nodes: Sequence[CollaboratorToolsNode]
  ) -> StateGraph:
    """add workflow edges"""
    tools_nodes_dict = {}
    router = Router()

    for tools_node in tools_nodes:
      if not tools_node.conditional_edge_node:
        continue
      workflow.add_conditional_edges(
        tools_node.name,
        router.router,
        {
          "continue": tools_node.conditional_edge_node,
          CALL_TOOL_NODE_NAME: CALL_TOOL_NODE_NAME,
          "end": END
        }
      )
      tools_nodes_dict[tools_node.name] = tools_node.name

    workflow.add_conditional_edges(
      CALL_TOOL_NODE_NAME,
      lambda x: x["sender"],
      tools_nodes_dict)

    return workflow


  def _set_entry_point(
    self,
    workflow: StateGraph,
    tools_nodes: List[CollaboratorToolsNode]
  ) -> StateGraph:
    """add workflow edges"""
    entry_point = tools_nodes[0].name
    for tools_node in tools_nodes:
      if tools_node.entrypoint_flag:
        entry_point = tools_node.name
        break
    workflow.set_entry_point(entry_point)
    return workflow


  def _create_tools_agent(
    self,
    llm: ChatAnthropicTools,
    tools: List[BaseTool],
    system_prompt: str
  ) -> RunnableSequence:
    """create tools agent"""

    prompt = ChatPromptTemplate.from_messages([
      ("system", TOOLS_PROMPT,),
      MessagesPlaceholder(variable_name = "messages"),])

    llm.bind_tools(tools)

    prompt = prompt.partial(system_message = system_prompt)
    prompt = prompt.partial(tool_names = ", ".join([tool.name for tool in tools]))
    return prompt | llm


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    self.workflow = StateGraph(AgentState)
    self.workflow = self._add_workflow_nodes(
      model_name = self.model_name,
      workflow = self.workflow,
      tools_nodes = self.tools_nodes,
    )
    self.workflow = self._add_workflow_edges(
      workflow = self.workflow,
      tools_nodes = self.tools_nodes,
    )
    self.workflow = self._set_entry_point(
      workflow = self.workflow,
      tools_nodes = self.tools_nodes,
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
  def from_tools_nodes(
    cls,
    tools_nodes: Sequence[CollaboratorToolsNode],
    model_name: Optional[str] = "claude-3-haiku-20240307",
    recursion_limit: Optional[int] = 100,
  ) -> "Collaborator":
    """Construct a Collaborator from an LLM and tools."""

    return cls(
      model_name = model_name,
      recursion_limit = recursion_limit,
      tools_nodes = tools_nodes,
    )
