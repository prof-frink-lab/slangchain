"""Schemas"""
from typing import (
  Annotated,
  List,
  Sequence,
  TypedDict,
  Optional,
  Union,
  Callable,
)
from typing_extensions import TypedDict
import operator

from playwright.async_api import Page

from langchain_core.messages import BaseMessage
from langchain.tools.base import (
  BaseTool, StructuredTool
)

from pydantic import BaseModel, Extra, Field


class ToolsNode(BaseModel):
  """ToolsNode"""
  name: str
  tools: Sequence[BaseTool]
  prompt: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @classmethod
  def from_objs(
    cls,
    name: str,
    tools: Sequence[Union[BaseTool, StructuredTool]],
    prompt: str):
    """from_objs"""
    return cls(
      name = name,
      tools = tools,
      prompt = prompt
    )


class AgentSupervisorAgentState(TypedDict):
  """
  The annotation tells the graph that new messages will always
  be added to the current states"""
  messages: Annotated[Sequence[BaseMessage], operator.add]
  next: str


class BBox(TypedDict):
  """BBox"""
  x: float
  y: float
  text: str
  type: str
  ariaLabel: str


class Prediction(TypedDict):
  """Prediction"""
  action: str
  args: Optional[List[str]]


class WebNavigationAgentState(TypedDict):
  """WebNavigationAgentState"""
  page: Page  # The Playwright web page lets us interact with the web environment
  input: str  # User request
  img: str  # b64 encoded screenshot
  bboxes: List[BBox]  # The bounding boxes from the browser annotation function
  prediction: Prediction  # The Agent's output
  # A system message (or messages) containing the intermediate steps
  scratchpad: List[BaseMessage]
  observation: str  # The most recent response from a tool


class CollaboratorAgentState(TypedDict):
  """CollaboratorAgentState"""
  messages: Annotated[Sequence[BaseMessage], operator.add]
  sender: str


class CollaboratorToolsNode(ToolsNode):
  """CollaboratorToolsNode"""
  entrypoint_flag: Optional[bool] = Field(default = False)
  conditional_edge_node: Optional[str] = Field(default = None)

  @classmethod
  def from_objs(
    cls,
    tools: Sequence[Union[BaseTool, StructuredTool]],
    name: str,
    prompt: str,
    entrypoint_flag: Optional[bool] = False,
    conditional_edge_node: Optional[str] = None):
    """from_objs"""
    return cls(
      tools = tools,
      name = name,
      prompt = prompt,
      entrypoint_flag = entrypoint_flag,
      conditional_edge_node = conditional_edge_node
    )

class AgentTeamToolsNode(ToolsNode):
  """AgentTeamToolsNode"""
  prelude: Optional[Callable] = Field(default = None)

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @classmethod
  def from_objs(
    cls,
    name: str,
    tools: Sequence[Union[BaseTool, StructuredTool]],
    prompt: str,
    prelude: Optional[Callable] = None):
    """from_objs"""
    return cls(
      tools = tools,
      name = name,
      prompt = prompt,
      prelude = prelude
    )


class AgentTeam(BaseModel):
  """Hierarchical Agent Team"""
  name: str
  agent_state: type
  tools_nodes: Sequence[AgentTeamToolsNode]

  @classmethod
  def from_objs(
    cls,
    name: str,
    agent_state: type,
    tools_nodes: Sequence[AgentTeamToolsNode],
  ):
    """from_objs"""
    return cls(
      name = name,
      agent_state = agent_state,
      tools_nodes = tools_nodes,
    )


class HierarchicalAgentState(TypedDict):
  """"Hierarchical Agent State"""
  messages: Annotated[List[BaseMessage], operator.add]
  next: str
