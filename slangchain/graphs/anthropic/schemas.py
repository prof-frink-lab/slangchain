"""Schemas"""
from typing import (
  Annotated, List, Sequence, TypedDict, Optional
)
import operator

from playwright.async_api import Page

from langchain.tools.base import BaseTool
from langchain_core.messages import BaseMessage

from pydantic import BaseModel, Extra


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
    tool: BaseTool,
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
  x: float
  y: float
  text: str
  type: str
  ariaLabel: str


class Prediction(TypedDict):
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
