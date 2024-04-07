"""Agent Supervisor class"""
from typing import (
  Dict, Sequence, List, Optional, Any
)
import logging
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_anthropic.output_parsers import ToolsOutputParser

from langchain.tools import BaseTool

from langchain.callbacks.manager import (
  Callbacks,
  CallbackManagerForToolRun
)
from langchain.agents import AgentExecutor
from langchain.chains.base import Chain

from langgraph.pregel import Pregel
from langgraph.graph import StateGraph, END

from slangchain.llms.chat_anthropic_tools import ChatAnthropicTools

from slangchain.agents.format_scratchpad.chat_anthropic_functions import (
    format_to_anthropic_function_messages,
)
from slangchain.graphs.anthropic.schemas import (
  NodeTool,
  AgentSupervisorAgentState as AgentState
)
from slangchain.agents.output_parsers.chat_anthropic_tools import (
  ChatAnthropicToolsAgentOutputParser
)

NEXT = "next"
FINISH = "FINISH"
SUPERVISOR = "supervisor"
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status."
    " Be very economical and use the minimum workers to achieve your task."
    " When finished, respond with FINISH."
    "Given the conversation below, who should act next?"
    " Or should we FINISH? Select one of: {options}"
    " by executing the tool"
)


logger = logging.getLogger(__name__)

def agent_node(state, agent, name):
  """agent node"""
  result = agent.invoke(state)
  return {"messages": [AIMessage(content=result["output"], name=name)]}


class RouteSchema(BaseModel):
  """RouteSchema"""
  next: str

  @classmethod
  def from_next_values(cls, next_values: List[str]):
    """Set the enum for the next field based on the provided values"""
    cls.__fields__["next"].field_info.extra["enum"] = next_values
    return cls(next=next_values[0])


class RouteTool(BaseTool):
  """RouteTool"""
  name = "route"
  description = "Select the next role."
  args_schema = RouteSchema

  def __init__(self, options: List[str], **kwargs):
    super().__init__(**kwargs)
    RouteSchema.from_next_values(options)  # Set enum for next field
    self.args_schema = RouteSchema

  def _run(
    self,
    next: str,
    run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
    """Implement the functionality of the tool here"""
    return { "next": next }


class AgentSupervisor(Chain):
  """AgentSupervisor"""
  model_name: str
  max_iterations: Optional[int] = Field(default = 15)
  return_intermediate_steps: bool = Field(default = False)
  early_stopping_method: Optional[str] = Field(default = "generate")
  recursion_limit: Optional[int] = Field(default = 100)
  callbacks: Optional[Callbacks] = Field(default = None)
  verbosity: Optional[bool] = Field(default = False)

  node_tools: Optional[List[NodeTool]]
  workflow: Optional[StateGraph] = Field(default = None)
  graph: Optional[Pregel] = Field(default = None)

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
    if not values["node_tools"]:
      return ValueError("node_tools list empty")
    if not all(isinstance(node_tool, NodeTool) for node_tool in values["node_tools"]):
      raise TypeError("tools all must be of BaseTool type")
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


  def _create_tools_agent(
    self,
    llm: ChatAnthropicTools,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate
  ) -> Runnable:
    """Create an agent that uses Anthropic tools. 
    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
      raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm.bind_tools(tools = tools)

    agent = (
      RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_anthropic_function_messages(
          x["intermediate_steps"]
        )
      )
      | prompt
      | llm
      | ChatAnthropicToolsAgentOutputParser(latest_message_only = True)
    )

    return agent


  def _create_tools_agent_executor(
    self,
    model_name: str,
    tools: list,
    system_prompt: str
  ) -> AgentExecutor:
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
      ]
    )

    llm = ChatAnthropicTools(model_name = model_name)
    agent = self._create_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
      agent = agent,
      tools = tools,
      max_iterations = self.max_iterations,
      return_intermediate_steps = self.return_intermediate_steps,
      verbose = self.verbosity,
      early_stopping_method = self.early_stopping_method,
      handle_parsing_errors = True,
      callbacks = self.callbacks,
    )

    return executor


  def _create_supervisor(
    self,
    model_name: str,
    node_tools: List[NodeTool]) -> Runnable:

    members = [ node_tool.name for node_tool in node_tools ]
    options = [ FINISH ] + members

    route_tool = RouteTool(options = options)


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(options=str(options), members=", ".join(members))

    llm = ChatAnthropicTools(model_name = model_name)
    llm.bind_tools([route_tool])

    supervisor_chain = (
      prompt
      | llm
      | ToolsOutputParser(args_only=True, first_tool_only=True)
    )

    return supervisor_chain


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    supervisor_chain = self._create_supervisor(self.model_name, self.node_tools)

    for node_tool in self.node_tools:
      agent = self._create_tools_agent_executor(
        self.model_name,
        [node_tool.tool],
        (
          f"{node_tool.description}"
          " Be very economical and use the minimum number of tool iterations to achieve your task.")
      )
      node = functools.partial(agent_node, agent = agent, name = node_tool.name)
      self.workflow.add_node(node_tool.name, node)

    self.workflow.add_node(SUPERVISOR, supervisor_chain)

    for node_tool in self.node_tools:
      self.workflow.add_edge(node_tool.name, SUPERVISOR)

    conditional_map = { node_tool.name: node_tool.name for node_tool in self.node_tools }
    conditional_map[FINISH] = END

    self.workflow.add_conditional_edges(SUPERVISOR, lambda x: x[NEXT], conditional_map)
    # Finally, add entrypoint
    self.workflow.set_entry_point(SUPERVISOR)

    return self.workflow


  def compile_graph(self) -> Pregel:
    """compile graph"""
    self.graph = self.workflow.compile()
    return self.graph


  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:

    message = inputs[self.input_key]
    self.workflow = StateGraph(AgentState)
    self.init_workflow_nodes()
    self.compile_graph()

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
  def from_node_tools(
    cls,
    node_tools: Sequence[NodeTool],
    max_iterations: Optional[int] = 15,
    model_name: Optional[str] = "claude-3-haiku-20240307",
    return_intermediate_steps: Optional[bool] = False,
    early_stopping_method: Optional[str] = "generate",
    recursion_limit: Optional[int] = 100,
    callbacks: Optional[Callbacks] = None,
    verbosity: Optional[bool] = False,
  ) -> "AgentSupervisor":
    """Construct an AgentSupervisor from tools."""

    return cls(
      model_name = model_name,
      recursion_limit = recursion_limit,
      max_iterations = max_iterations,
      return_intermediate_steps = return_intermediate_steps,
      early_stopping_method = early_stopping_method,
      node_tools = node_tools,
      callbacks = callbacks,
      verbosity = verbosity,
    )
