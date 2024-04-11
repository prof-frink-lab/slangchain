"""Hierarchical Agent Teams class"""
from typing import (
  Dict, Sequence, List, Optional, Any
)
import logging
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_anthropic.output_parsers import ToolsOutputParser

from langchain.tools import BaseTool

from langchain.callbacks.manager import (
  Callbacks,
)
from langchain.agents import AgentExecutor
from langchain.chains.base import Chain

from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from slangchain.llms.chat_anthropic_tools import ChatAnthropicTools

from slangchain.graphs.anthropic.schemas import (
  AgentTeamToolsNode,
  AgentTeam,
  HierarchicalAgentState as AgentState
)

from slangchain.agents.format_scratchpad.chat_anthropic_functions import (
  format_to_anthropic_function_messages,
)
from slangchain.agents.output_parsers.chat_anthropic_tools import (
  ChatAnthropicToolsAgentOutputParser,
)

logger = logging.getLogger(__name__)

NEXT = "next"
FINISH = "FINISH"
SUPERVISOR = "supervisor"
TEAM_SUPERVISOR_PROMPT = (
  "You are a supervisor tasked with managing a conversation between the"
  " following workers:  {team_members}. Given the following user request,"
  " respond with the worker to act next. Each worker will perform a"
  " task and respond with their results and status. When finished,"
  " respond with FINISH."
  "\nGiven the conversation below, who should act next?"
  " Or should we FINISH? Select one of: {options}")
AGENT_TEAM_PROMPT = (
  "\nWork autonomously according to your specialty, using the tools available to you."
  " Do not ask for clarification. Your other team members (and other teams)"
  " will collaborate with you with their own specialties."
  " You are chosen for a reason! You are one of the following team members: {team_members}.")
SUPERVISOR = "supervisor"



class AgentNode:
  """AgentNode"""
  def agent_node(self, state, agent, name):
    """agent node"""
    result = agent.invoke(state)
    logger.debug("Agent Node: %s | %s\n\n", name, result["output"])
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


class HierarchicalAgentTeams(Chain):
  """HierarchicalAgentTeams"""
  model_name: str
  agent_teams: Sequence[AgentTeam]

  max_iterations: Optional[int] = Field(default = 15)
  return_intermediate_steps: bool = Field(default = False)
  early_stopping_method: Optional[str] = Field(default = "generate")
  recursion_limit: Optional[int] = Field(default = 100)
  callbacks: Optional[Callbacks] = Field(default = None)
  verbosity: Optional[bool] = Field(default = False)
  workflow: Optional[StateGraph] = Field(default = None)

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
    if not values["agent_teams"]:
      return ValueError("agent_teams list empty")
    if not all(isinstance(agent_team, AgentTeam) \
               for agent_team in values["agent_teams"]):
      raise TypeError("tools all must be of AgentTeam type")
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


  def _create_agent_team_executor(
    self,
    llm: ChatAnthropicTools,
    tools: Sequence[BaseTool],
    system_prompt: str
  ) -> AgentExecutor:
    # Each worker node will be given a name and some tools.
    system_prompt += AGENT_TEAM_PROMPT

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


  def _create_team_supervisor(
    self,
    model_name: str,
    members: Sequence[str]) -> Runnable:

    options = [ FINISH ] + members

    route_tool =  {
      "name": "route",
      "description": "Select the next role.",
      "input_schema": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
          "next": {
            "title": "Next",
            "anyOf": [
              {"enum": options},
            ],
          },
        },
        "required": ["next"],
      },
    }
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", TEAM_SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
      ]
    ).partial(options = str(options), team_members = ", ".join(members))


    llm = ChatAnthropicTools(model_name = model_name)
    llm.bind_tools([route_tool])

    supervisor_chain = (
      prompt
      | llm
      | ToolsOutputParser(args_only=True, first_tool_only=True)
    )

    return supervisor_chain


  def _create_team_tools_node(
    self,
    agent_node: AgentNode,
    llm: ChatAnthropicTools,
    tools_node: AgentTeamToolsNode) -> functools.partial:
    """create team tools node"""

    agent = self._create_agent_team_executor(
      llm = llm,
      tools = tools_node.tools,
      system_prompt = tools_node.prompt
    )
    if tools_node.prelude:
      agent = tools_node.prelude | agent

    node = functools.partial(
        agent_node.agent_node, agent = agent, name = tools_node.name
    )
    return node


  def _create_agent_team_graph(
    self,
    model_name: str,
    agent_team: AgentTeam
    ) -> Pregel:
    """Create agent team chain"""
    tools_nodes_dict = {}
    workflow = StateGraph(agent_team.agent_state)
    agent_node = AgentNode()

    supervisor = self._create_team_supervisor(
      model_name = model_name,
      members = [ team_node.name for team_node in agent_team.tools_nodes ]
    )
    workflow.add_node(SUPERVISOR, supervisor)

    for tools_node in agent_team.tools_nodes:
      team_tools_node = self._create_team_tools_node(
        agent_node = agent_node,
        llm = ChatAnthropicTools(model_name = model_name),
        tools_node = tools_node
      )
      workflow.add_node(tools_node.name, team_tools_node)
      workflow.add_edge(tools_node.name, SUPERVISOR)
      tools_nodes_dict[tools_node.name] = tools_node.name


    tools_nodes_dict[FINISH] = END

    workflow.add_conditional_edges(
      SUPERVISOR,
      lambda x: x[NEXT],
      tools_nodes_dict
    )
    workflow.set_entry_point(SUPERVISOR)

    graph = workflow.compile()

    return graph


  def _create_agent_team_chain(
    self,
    model_name: str,
    agent_team: AgentTeam) -> RunnableSequence:
    """Create agent team chain"""

    agent_team_chain = None
    graph = self._create_agent_team_graph(model_name=model_name, agent_team=agent_team)
    members = [tools_node.name for tools_node in agent_team.tools_nodes]
    agent_team_chain = functools.partial(self._enter_chain, team_members = members) | graph

    return agent_team_chain


  def _get_last_message(self, state: AgentState) -> str:
    return state["messages"][-1].content


  def _join_graph(self, response: Dict[str, Any]):
    return {"messages": [response["messages"][-1]]}


  def _enter_chain(
    self,
    message: str,
    team_members: List[str]) -> Dict[str, Any]:
    """Enter chain function"""
    logger.debug("Enter chain: %s | %s\n", message, team_members)
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(team_members)
    }

    return results


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    tools_nodes_dict = {}
    self.workflow = StateGraph(AgentState)


    supervisor = self._create_team_supervisor(
      model_name = self.model_name,
      members = [ agent_team.name for agent_team in self.agent_teams ]
    )
    self.workflow.add_node(SUPERVISOR, supervisor)

    for agent_team in self.agent_teams:
      agent_team_chain = self._create_agent_team_chain(
        model_name = self.model_name,
        agent_team = agent_team
      )

      self.workflow.add_node(
        agent_team.name, self._get_last_message | agent_team_chain | self._join_graph)

      self.workflow.add_edge(agent_team.name, SUPERVISOR)
      tools_nodes_dict[agent_team.name] = agent_team.name

    tools_nodes_dict[FINISH] = END

    self.workflow.add_conditional_edges(
        SUPERVISOR,
        lambda x: x[NEXT],
        tools_nodes_dict,
    )
    self.workflow.set_entry_point(SUPERVISOR)

    return self.workflow


  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:
    message = inputs[self.input_key]

    self.workflow = self.init_workflow_nodes()
    graph = self.workflow.compile()

    result : Dict[str, Any] = {}

    for graph_stream in graph.stream(
      {
        "messages": [
            HumanMessage(content=message)
        ]
      },
      {"recursion_limit": self.recursion_limit},
    ):
      if "__end__" not in graph_stream:
        logger.info("\n%s", graph_stream)
        logger.info("----")
      result = graph_stream.get("__end__", {})

    return {self.output_key: result}


  @classmethod
  def from_agent_teams(
    cls,
    agent_teams: Sequence[AgentTeam],
    model_name: Optional[str] = "claude-3-haiku-20240307",
    max_iterations: Optional[int] = 15,
    return_intermediate_steps: Optional[bool] = False,
    early_stopping_method: Optional[str] = "generate",
    recursion_limit: Optional[int] = 100,
    callbacks: Optional[Callbacks] = None,
    verbosity: Optional[bool] = False,
  ) -> "HierarchicalAgentTeams":
    """Construct an HierarchicalAgentTeams from agent teams."""

    return cls(
      model_name = model_name,
      agent_teams = agent_teams,
      recursion_limit = recursion_limit,
      max_iterations = max_iterations,
      return_intermediate_steps = return_intermediate_steps,
      early_stopping_method = early_stopping_method,
      callbacks = callbacks,
      verbosity = verbosity,
    )
