"""Schemas"""
from typing import Annotated, Sequence, TypedDict
import operator

from langchain.tools.base import BaseTool
from langchain_core.messages import BaseMessage

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
    tool: BaseTool,
    name: str,
    description: str):
    """from_objs"""
    return cls(
      tool = tool,
      name = name,
      description = description
    )

class AgentState(TypedDict):
  """
  The annotation tells the graph that new messages will always
  be added to the current states"""
  messages: Annotated[Sequence[BaseMessage], operator.add]
  next: str
