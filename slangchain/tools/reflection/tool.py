"""Reflection"""
from typing import Optional, Type
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

class ReflectionInput(BaseModel):
  """Input for UrlImageUrlsDocLoader."""
  reflections: str = Field(
    description="The critique and reflections on the sufficiency, superfluency,"
    " and general quality of the response"
  )

  score: int = Field(
    description="Score from 0-10 on the quality of the candidate response.",
    gte=0,
    lte=10,
  )

  found_solution: bool = Field(
    description="Whether the response has fully solved the question or task."
  )

class ReflectionTool(BaseTool):
  """Reflection"""

  name: str = "Reflection"

  description: str = "tool that reflects and grades the assistant response"

  args_schema: Optional[Type[BaseModel]] = ReflectionInput

  def _run(
    self,
    url: str,
    query: Optional[str] = None,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""
    return
