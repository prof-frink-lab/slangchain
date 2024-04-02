"""Schemas"""
from typing import (
  Annotated, List, Sequence, TypedDict, Optional, Union
)
import operator

from playwright.async_api import Page

from langchain_core.messages import BaseMessage
from langchain.tools.base import (
  BaseTool, StructuredTool
)

from pydantic import BaseModel, Extra, Field


class NodeTool(BaseModel):
  """NodeTool"""
  tool: BaseTool
  name: str
  description: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @classmethod
  def from_objs(
    cls,
    tool: Union[BaseTool, StructuredTool],
    name: str,
    description: str):
    """from_objs"""
    return cls(
      tool = tool,
      name = name,
      description = description
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


class CollaboratorNodeTool(NodeTool):
  """CollaboratorNodeTool"""
  entrypoint_flag: Optional[bool] = Field(default = False)
  conditional_edge_node: Optional[str] = Field(default = None)

  @classmethod
  def from_objs(
    cls,
    tool: Union[BaseTool, StructuredTool],
    name: str,
    description: str,
    entrypoint_flag: Optional[bool] = False,
    conditional_edge_node: Optional[str] = None):
    """from_objs"""
    return cls(
      tool = tool,
      name = name,
      description = description,
      entrypoint_flag = entrypoint_flag,
      conditional_edge_node = conditional_edge_node
    )
