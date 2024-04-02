"""Anthropic Functions"""
import sys
import traceback
import json
from collections import defaultdict
from html.parser import HTMLParser
from typing import (
  Any,
  DefaultDict,
  Dict,
  List,
  Optional,
  cast
)
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
  BaseMessage,
  HumanMessage,
  AIMessage,
  SystemMessage
)
from langchain_core.runnables import RunnableConfig

from langchain_experimental.pydantic_v1 import root_validator

from langchain_anthropic import ChatAnthropic

from langchain.callbacks.manager import (
  CallbackManagerForLLMRun,
)
from langchain.schema import (
  ChatGeneration,
  ChatResult,
)


logger = logging.getLogger(__name__)

SYSTEM_FUNCTIONS_PROMPT = """In addition to responding, you must only use the following tools. \

{tools}

In order to use a tool, you must use <tool></tool> to specify the name, \
and the <tool_input></tool_input> tags to specify the parameters. \
Each parameter should be passed in as <$param_name>$value</$param_name>, \
Where $param_name is the name of the specific parameter, and $value \
is the value for that parameter.

You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that accepts a single \
parameter 'query' that could run a google search, in order to search \
for the weather in SF you would respond:

<tool>search</tool><tool_input><query>weather in SF</query></tool_input>
<observation>64 degrees</observation>"""


class TagParser(HTMLParser):
  """TagParser"""

  def __init__(self) -> None:
    """A heavy-handed solution, but it's fast for prototyping.

    Might be re-implemented later to restrict scope to the limited grammar, and
    more efficiency.

    Uses an HTML parser to parse a limited grammar that allows
    for syntax of the form:

      INPUT -> JUNK? VALUE*
      JUNK -> JUNK_CHARACTER+
      JUNK_CHARACTER -> whitespace | ,
      VALUE -> <IDENTIFIER>DATA</IDENTIFIER> | OBJECT
      OBJECT -> <IDENTIFIER>VALUE+</IDENTIFIER>
      IDENTIFIER -> [a-Z][a-Z0-9_]*
      DATA -> .*

    Interprets the data to allow repetition of tags and recursion
    to support representation of complex types.

    ^ Just another approximately wrong grammar specification.
    """
    super().__init__()

    self.parse_data: DefaultDict[str, List[Any]] = defaultdict(list)
    self.stack: List[DefaultDict[str, List[str]]] = [self.parse_data]
    self.success = True
    self.depth = 0
    self.data: Optional[str] = None


  def handle_starttag(self, tag: str, attrs: Any) -> None:
    """Hook when a new tag is encountered."""
    self.depth += 1
    self.stack.append(defaultdict(list))
    self.data = None


  def handle_endtag(self, tag: str) -> None:
    """Hook when a tag is closed."""
    self.depth -= 1
    top_of_stack = dict(self.stack.pop(-1))  # Pop the dictionary we don't need it

    # If a lead node
    is_leaf = self.data is not None
    # Annoying to type here, code is tested, hopefully OK
    value = self.data if is_leaf else top_of_stack
    # Difficult to type this correctly with mypy (maybe impossible?)
    # Can be nested indefinitely, so requires self referencing type
    self.stack[-1][tag].append(value)  # type: ignore
    # Reset the data so we if we encounter a sequence of end tags, we
    # don't confuse an outer end tag for belonging to a leaf node.
    self.data = None


  def handle_data(self, data: str) -> None:
    """Hook when handling data."""
    stripped_data = data.strip()
    # The only data that's allowed is whitespace or a comma surrounded by whitespace
    if self.depth == 0 and stripped_data not in (",", ""):
      # If this is triggered the parse should be considered invalid.
      self.success = False
    if stripped_data:  # ignore whitespace-only strings
      self.data = stripped_data


def _destrip(tool_input: Any) -> Any:
  """de-strip Dict"""
  if isinstance(tool_input, dict):
    return {k: _destrip(v) for k, v in tool_input.items()}
  elif isinstance(tool_input, list):
    if isinstance(tool_input[0], str):
      if len(tool_input) == 1:
        return tool_input[0]
      else:
        raise ValueError
    elif isinstance(tool_input[0], dict):
      return [_destrip(v) for v in tool_input]
    else:
      raise ValueError
  else:
    raise ValueError


class ChatAnthropicFunctions(BaseChatModel):
  """ChatAnthropicFunctions"""
  llm: BaseChatModel

  @root_validator(pre=True)
  def validate_environment(cls, values: Dict) -> Dict:
    """validate environment"""
    values["llm"] = values.get("llm") or ChatAnthropic(**values)
    return values

  @property
  def model(self) -> BaseChatModel:
    """For backwards compatibility."""
    return self.llm

  def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
  ) -> ChatResult:
    forced = False
    function_call = ""
    logger.debug("============= _generate ==============")
    logger.debug("kwargs: %s\n", kwargs)
    if "functions" in kwargs:
      # get the function call method
      if "function_call" in kwargs:
        function_call = kwargs["function_call"]
        del kwargs["function_call"]
      else:
        function_call = "auto"

      # should function calling be used
      if function_call != "none":
        content = SYSTEM_FUNCTIONS_PROMPT.format(tools=json.dumps(kwargs["functions"], indent=2))

        for message in messages:
          if isinstance(message, SystemMessage):
            if content not in message.content:
              message.content += "\n" + content
            break

      # is the function call a dictionary (forced function calling)
      if isinstance(function_call, dict):
        forced = True
        function_call_name = function_call["name"]
        messages.append(AIMessage(content=f"<tool>{function_call_name}</tool>"))

      del kwargs["functions"]
      if stop is None:
        stop = ["</tool_input>"]
      else:
        stop.append("</tool_input>")
    else:
      if "function_call" in kwargs:
        raise ValueError(
          "if `function_call` provided, `functions` must also be"
        )

    messages = self._format_messages(messages)

    logger.debug("formatted last message: \n %s", json.dumps(messages[-1].dict(), indent=2))
    response = None
    try:
      response = self.model.invoke(
        input = messages,
        stop = stop,
        config=RunnableConfig(callbacks=run_manager),
        **kwargs)

      logger.debug("response: %s\n", json.dumps(response.dict(), indent=2))

      completion = cast(str, response.content)
      if forced:
        tag_parser = TagParser()

        if "<tool_input>" in completion:
          tag_parser.feed(completion.strip() + "</tool_input>")
          v1 = tag_parser.parse_data["tool_input"][0]
          arguments = json.dumps(_destrip(v1))
        else:
          v1 = completion
          arguments = ""

        kwargs = {
          "function_call": {
            "name": function_call_name,
            "arguments": arguments,
          }
        }
        message = AIMessage(content="function_call", additional_kwargs=kwargs)

      elif "<tool>" in completion:
        logger.debug("***tool in completion***")
        tag_parser = TagParser()
        tag_parser.feed(completion.strip() + "</tool_input>")
        msg = completion.split("<tool>")[0].strip()
        v1 = tag_parser.parse_data["tool_input"][0]
        logger.debug("v1: %s", v1)
        kwargs = {
          "function_call": {
            "name": tag_parser.parse_data["tool"][0],
            "arguments": json.dumps(_destrip(v1)),
          }
        }
        if not msg:
          msg = "function_call"
        message = AIMessage(content=msg, additional_kwargs=kwargs)
      else:
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
        "error messages: \n %s",
        json.dumps(messages[-1].dict(), indent=2))
      logger.error("error response: %s", response)
      logger.error("=============================\n")
      traceback.print_exc()
      message = AIMessage(
        content= (
          f"Exception - Type: {ex_type}, Exception: {ex_value},"
          f" Traceback: {ex_traceback}, File: {filename}, Line: {line_number}"))

    logger.debug("response message: %s | %s\n", message.type, message)

    return ChatResult(generations=[ChatGeneration(message=message)])



  # HACK - this method recreates alternating message types required by Claude API.
  # https://docs.anthropic.com/claude/reference/migrating-from-text-completions-to-messages#:~:text=%F0%9F%92%A1-,Role%20names,-The%20Text%20Completions
  def _format_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
    """format messages"""

    formatted_messages: List[BaseMessage] = []

    previous_message = None

    for message in messages:
      if previous_message is None or isinstance(message, SystemMessage):
        formatted_messages.append(message)
        previous_message = message
        continue

      current_message = None

      if isinstance(message, type(previous_message)):

        if isinstance(message, AIMessage):
          current_message = HumanMessage(
            content = message.content,
            additional_kwargs = message.additional_kwargs)
        elif isinstance(message, HumanMessage):
          current_message = AIMessage(
            content = message.content,
            additional_kwargs = message.additional_kwargs)
        if current_message:
          formatted_messages.append(current_message)
      else:
        current_message = message
        formatted_messages.append(current_message)

      previous_message = current_message

    return formatted_messages


  @property
  def _llm_type(self) -> str:
    return "chat_anthropic_functions"
