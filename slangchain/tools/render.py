"""Anthropic Render"""
from typing import Type, Optional

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
from langchain_core.tools import BaseTool

from langchain_community.utils.openai_functions import (
  FunctionDescription,
  ToolDescription
)


def convert_pydantic_to_openai_function(
  model: Type[BaseModel],
  *,
  name: Optional[str] = None,
  description: Optional[str] = None,
) -> FunctionDescription:
  """Converts a Pydantic model to a function description for the OpenAI API."""
  schema = dereference_refs(model.schema())
  schema.pop("definitions", None)
  return {
    "name": name or schema["title"],
    "description": description or schema["description"],
    "parameters": schema,
  }


def format_tool_to_anthropic_function(tool: BaseTool) -> FunctionDescription:
  """Format tool into the OpenAI function API."""
  if tool.args_schema:
    return convert_pydantic_to_openai_function(
      tool.args_schema, name=tool.name, description=tool.description
    )
  else:
    return {
      "name": tool.name,
      "description": tool.description,
      "parameters": {
        # This is a hack to get around the fact that some tools
        # do not expose an args_schema, and expect an argument
        # which is a string.
        "properties": {
          "__arg1": {"title": "__arg1", "type": "string"},
        },
        "required": ["__arg1"],
        "type": "object",
      },
    }


def format_tool_to_anthropic_tool(tool: BaseTool) -> ToolDescription:
  """Format tool into the OpenAI function API."""
  function = format_tool_to_anthropic_function(tool)
  return {"type": "function", "function": function}
