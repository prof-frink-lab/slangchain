"""Agent Supervisor class"""
from typing import Dict, Sequence, List, Optional, Any
import logging
import functools
import copy 

from langchain.callbacks.manager import (
  Callbacks
)
from langchain.agents import AgentExecutor

from langchain.chains.base import Chain
from langchain.tools.base import BaseTool
from langgraph.pregel import Pregel
from langgraph.graph import StateGraph, END

from langchain_core.runnables.config import (
  RunnableConfig
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun

from slangchain.llms.chat_anthropic_functions import ChatAnthropicFunctions

from slangchain.agents.format_scratchpad.chat_anthropic_functions import (
    format_to_anthropic_function_messages,
)
from slangchain.tools.render import format_tool_to_anthropic_function

from slangchain.agents.output_parsers.chat_anthropic_functions import (
  ChatAnthropicFunctionsAgentOutputParser,
  JsonOutputFunctionsParser
)
from slangchain.graphs.anthropic.schemas import (
  NodeTool,
  AgentSupervisorAgentState as AgentState
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
)


logger = logging.getLogger(__name__)

def agent_node(state, agent, name):
  """agent node"""
  result = agent.invoke(state)
  return {"messages": [AIMessage(content=result["output"], name=name)]}

class AgentSupervisor(Chain):
  """AgentSupervisor"""
  llm: ChatAnthropicFunctions
  tools: Sequence[BaseTool]
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
    if not isinstance(values["llm"], ChatAnthropicFunctions):
      raise TypeError("llm must be of instance ChatAnthropicFunctions")
    if not values["tools"]:
      return ValueError("tools list empty")
    if not all(isinstance(tool, BaseTool) for tool in values["tools"]):
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


  def _create_functions_agent(
    self,
    llm: ChatAnthropicFunctions,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate
  ) -> Runnable:
    """Create an agent that uses OpenAI tools. 
    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
      raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_fns = llm.bind(
      functions = [ format_tool_to_anthropic_function(tool) for tool in tools ],
      # function_call = { "name": tools[0].name }
      function_call = tools[0].name
    )

    agent = (
      RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_anthropic_function_messages(
          x["intermediate_steps"]
        )
      )
      | prompt
      | llm_with_fns
      | ChatAnthropicFunctionsAgentOutputParser()
    )
    return agent


  def _create_tools_agent(
    self,
    llm: ChatAnthropicFunctions,
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

    agent = self._create_functions_agent(llm, tools, prompt)
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
    llm: ChatAnthropicFunctions,
    node_tools: List[NodeTool]) -> Runnable:

    members = [ node_tool.name for node_tool in node_tools ]
    options = [ FINISH ] + members

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
      prompt
      | llm.bind(functions=[function_def], function_call="route")
      | JsonOutputFunctionsParser()
    )

    return supervisor_chain


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    supervisor_chain = self._create_supervisor(self.llm, self.node_tools)

    for node_tool in self.node_tools:
      agent = self._create_tools_agent(
        self.llm,
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

    _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
    message = inputs[self.input_key]
    self.init_workflow_nodes()
    self.compile_graph()
    config = RunnableConfig(recursion_limit = self.recursion_limit)

    result : Dict[str, Any] = {}

    if self.verbosity:
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
    else:
      result = self.graph.invoke(
        input = {"messages": [HumanMessage(content=message)]},
        config = config)

    return {self.output_key: result}


  @classmethod
  def from_llm_and_tools(
    cls,
    llm: ChatAnthropicFunctions,
    tools: Sequence[BaseTool],
    max_iterations: Optional[int] = 15,
    return_intermediate_steps: Optional[bool] = False,
    early_stopping_method: Optional[str] = "generate",
    recursion_limit: Optional[int] = 100,
    callbacks: Optional[Callbacks] = None,
    verbosity: Optional[bool] = False,
  ) -> "AgentSupervisor":
    """Construct an AgentSupervisor from an LLM and tools."""

    if not isinstance(llm, ChatAnthropicFunctions):
      raise ValueError("Only supported with ChatAnthropicFunctions models.")

    node_tools = [ NodeTool.from_objs(
      tool = tool, name = tool.name, description = tool.description) for tool in tools ]

    workflow = StateGraph(AgentState)

    return cls(
      llm = llm,
      tools = tools,
      recursion_limit = recursion_limit,
      max_iterations = max_iterations,
      return_intermediate_steps = return_intermediate_steps,
      early_stopping_method = early_stopping_method,
      node_tools = node_tools,
      workflow = workflow,
      callbacks = callbacks,
      verbosity = verbosity,
    )
