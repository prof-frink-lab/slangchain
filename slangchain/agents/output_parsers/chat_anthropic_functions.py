"""Anthropic functions"""
from typing import List, Union, Any, Optional
import json
from json import JSONDecodeError
import jsonpatch
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
  AIMessage,
  BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
)
from langchain.agents.agent import AgentOutputParser
from langchain.output_parsers.json import parse_partial_json

class ChatAnthropicFunctionsAgentOutputParser(AgentOutputParser):
  """Parses a message into agent action/finish.

  Is meant to be used with OpenAI models, as it relies on the specific
  function_call parameter from OpenAI to convey what tools to use.

  If a function_call parameter is passed, then that is used to get
  the tool and tool input.

  If one is not passed, then the AIMessage is assumed to be the final output.
  """

  @property
  def _type(self) -> str:
    return "anthropic-functions-agent"

  @staticmethod
  def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
      raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
      function_name = function_call["name"]
      try:
        if len(function_call["arguments"].strip()) == 0:
          # OpenAI returns an empty string for functions containing no args
          _tool_input = {}
        else:
          # otherwise it returns a json object
          _tool_input = json.loads(function_call["arguments"], strict=False)
      except JSONDecodeError:
        raise OutputParserException(
          f"Could not parse tool input: {function_call} because "
          f"the `arguments` is not valid JSON."
        )

      # HACK HACK HACK:
      # The code that encodes tool input into Open AI uses a special variable
      # name called `__arg1` to handle old style tools that do not expose a
      # schema and expect a single string argument as an input.
      # We unpack the argument here if it exists.
      if "__arg1" in _tool_input:
        tool_input = _tool_input["__arg1"]
      else:
        tool_input = _tool_input

      content_msg = f"responded: {message.content}\n" if message.content else "\n"
      log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
      return AgentActionMessageLog(
        tool=function_name,
        tool_input=tool_input,
        log=log,
        message_log=[message],
      )

    return AgentFinish(
      return_values={"output": message.content}, log=str(message.content)
    )

  def parse_result(
    self, result: List[Generation], *, partial: bool = False
  ) -> Union[AgentAction, AgentFinish]:
    if not isinstance(result[0], ChatGeneration):
      raise ValueError("This output parser only works on ChatGeneration output")

    message = result[0].message
    return self._parse_ai_message(message)

  def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
    raise ValueError("Can only parse messages")


class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
  """Parse an output as the Json object."""

  strict: bool = False
  """Whether to allow non-JSON-compliant strings.
  
  See: https://docs.python.org/3/library/json.html#encoders-and-decoders
  
  Useful when the parsed output may include unicode characters or new lines.
  """

  args_only: bool = True
  """Whether to only return the arguments to the function call."""

  @property
  def _type(self) -> str:
    return "json_functions"

  def _diff(self, prev: Optional[Any], next: Any) -> Any:
    return jsonpatch.make_patch(prev, next).patch

  def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
    if len(result) != 1:
      raise OutputParserException(
        f"Expected exactly one result, but got {len(result)}"
      )
    generation = result[0]
    if not isinstance(generation, ChatGeneration):
      raise OutputParserException(
        "This output parser can only be used with a chat generation."
      )
    message = generation.message

    if not message.additional_kwargs:
      message.additional_kwargs={
        'function_call': {'arguments': '{"next":"FINISH"}', 'name': 'route'}}
    try:
      function_call = message.additional_kwargs["function_call"]
    except KeyError as exc:
      if partial:
        return None
      else:
        raise OutputParserException(f"Could not parse function call: {exc}")
    try:
      if partial:
        try:
          if self.args_only:
            return parse_partial_json(
              function_call["arguments"], strict=self.strict
            )
          else:
            return {
              **function_call,
              "arguments": parse_partial_json(
                function_call["arguments"], strict=self.strict
              ),
            }
        except json.JSONDecodeError:
          return None
      else:
        if self.args_only:
          try:
            return json.loads(
              function_call["arguments"], strict=self.strict
            )
          except (json.JSONDecodeError, TypeError) as exc:
            raise OutputParserException(
              f"Could not parse function call data: {exc}"
            )
        else:
          try:
            return {
              **function_call,
              "arguments": json.loads(
                function_call["arguments"], strict=self.strict
              ),
            }
          except (json.JSONDecodeError, TypeError) as exc:
            raise OutputParserException(
              f"Could not parse function call data: {exc}"
            )
    except KeyError:
      return None

  # This method would be called by the default implementation of `parse_result`
  # but we're overriding that method so it's not needed.
  def parse(self, text: str) -> Any:
    raise NotImplementedError()
