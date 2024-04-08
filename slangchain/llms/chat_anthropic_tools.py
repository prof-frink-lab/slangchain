"""Anthropic Functions"""
import sys
import traceback
import json

from typing import (
  Any,
  Dict,
  List,
  Optional,
  Sequence,
  Union,
  Literal,
  Callable,
  Type,
  cast
)

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
  BaseMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage
)
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable

from langchain_experimental.pydantic_v1 import BaseModel, Field, root_validator

from langchain_anthropic import ChatAnthropic

from langchain.callbacks.manager import (
  CallbackManagerForLLMRun,
)
from langchain.schema import (
  ChatGeneration,
  ChatResult,
)
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


class ChatAnthropicTools(BaseChatModel):
  """ChatAnthropicTools"""
  llm: BaseChatModel
  llm_with_tools: Optional[Runnable[LanguageModelInput, BaseMessage]] = Field(default = None)

  @root_validator(pre=True)
  def validate_environment(cls, values: Dict) -> Dict:
    """validate environment"""
    values["llm"] = values.get("llm") or ChatAnthropic(**values)
    return values

  @property
  def model(self) -> BaseChatModel:
    """For backwards compatibility."""
    if self.llm_with_tools:
      return self.llm_with_tools
    return self.llm

  def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
  ) -> ChatResult:

    logger.debug(
      "messages: \n %s\n\n", messages)

    messages = self._format_messages(messages)

    logger.debug(
      "formatted messages: \n %s", messages)
    response = None
    try:

      response = self.model.invoke(
        input = messages,
        stop = stop,
        config=RunnableConfig(callbacks=run_manager),
        **kwargs)

      logger.debug("response: %s\n", json.dumps(response.dict(), indent=2))

      if isinstance(response.content, str):
        response.content = cast(str, response.content).strip()

      if not response.content:
        raise ValueError("response is empty or null")
      message = response
    except Exception:
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      logger.error("=============================\n")
      logger.error(
        "error messages: \n %s", messages)
      logger.error("error response: %s", response)
      logger.error("=============================\n")
      traceback.print_exc()
      message = AIMessage(
        content= (
          f"Exception - Type: {ex_type}, Exception: {ex_value},"
          f" Traceback: {ex_traceback}, File: {filename}, Line: {line_number}"))

    logger.debug("response message: %s | %s\n", message.type, message)

    return ChatResult(generations=[ChatGeneration(message=message)])


  def bind_tools(
    self,
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
    *,
    tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
    **kwargs: Any,
  ) -> Runnable[LanguageModelInput, BaseMessage]:
    """Bind tool-like objects to this chat model."""
    self.llm_with_tools = self.llm.bind_tools(tools)
    return self.llm_with_tools


  # HACK - this method recreates alternating message types required by Claude API.
  # https://docs.anthropic.com/claude/reference/migrating-from-text-completions-to-messages#:~:text=%F0%9F%92%A1-,Role%20names,-The%20Text%20Completions
  def _format_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
    """format messages"""

    formatted_messages: List[BaseMessage] = []

    previous_message = None

    current_message = None

    for message in messages:
      if previous_message is None or isinstance(message, SystemMessage):
        formatted_messages.append(message)
        previous_message = message
        continue

      current_message = None
      # Claude's tool binding API expects the next message after a tool_use to be a ToolMessage.
      # Otherwise, the model will return a bad request error.
      if isinstance(previous_message.content, list) and previous_message.content:
        if previous_message.content[-1].get("type") == "tool_use":
          current_message = ToolMessage(
            content = message.content,
            additional_kwargs = message.additional_kwargs,
            tool_call_id = previous_message.content[-1].get("id"),
            name = message.name)
        else:
          current_message = message
      elif isinstance(message, type(previous_message)):
        if isinstance(message, AIMessage):
          current_message = HumanMessage(
            content = message.content,
            additional_kwargs = message.additional_kwargs,
            name = message.name)
        elif isinstance(message, HumanMessage):
          current_message = AIMessage(
            content = message.content,
            additional_kwargs = message.additional_kwargs,
            name = message.name)
        else:
          current_message = message
      else:
        current_message = message

      formatted_messages.append(current_message)
      previous_message = current_message

    # Functionality to address Claude's tool binding constraint.
    # The Claude API returns the error below if the last message is an AIMessage:
    # "Your API request included an `assistant` message in the final position,
    # which would pre-fill the `assistant` response.
    # When using tools, pre-filling the `assistant` response is not supported."
    if isinstance(current_message, AIMessage) and isinstance(self.model, Runnable):
      if isinstance(formatted_messages[-2].content, str):
        previous_content = [
        {
          "type": "text",
          "text": formatted_messages[-2].content
        },
        {
          "type": "text",
          "text": current_message.content
        }]
        formatted_messages[-2].content = previous_content
      elif isinstance(formatted_messages[-2].content, list):
        formatted_messages[-2].content.append(
          {
            "type": "text",
            "text": current_message.content
          })
      formatted_messages = formatted_messages[:-1]
    return formatted_messages


  @property
  def _llm_type(self) -> str:
    return "chat_anthropic_tools"
