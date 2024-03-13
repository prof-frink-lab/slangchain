"""XML parser file"""
from typing import Union
import logging
import json
from json import JSONDecodeError
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

logger = logging.getLogger()

class XMLAgentOutputParser(AgentOutputParser):
  """Parses tool invocations and final answers in XML format.

  Expects output to be in one of two formats.

  If the output signals that an action should be taken,
  should be in the below format. This will result in an AgentAction
  being returned.

  ```
  <tool>search</tool>
  <tool_input>what is 2 + 2</tool_input>
  ```

  If the output signals that a final answer should be given,
  should be in the below format. This will result in an AgentFinish
  being returned.

  ```
  <final_answer>Foo</final_answer>
  ```
  """

  def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
    """parse function"""

    if "</observation>" in text:
      text = text.split("</observation>")[0]

    if "</tool>" in text:

      if len(text.split("</tool>")) == 2:
        tool, tool_input = text.split("</tool>")

        _tool = tool.split("<tool>")[1]

        _tool_input = tool_input.split("<tool_input>")

        if len(_tool_input) == 2:
          _tool_input = _tool_input[1]
        else:
          _tool_input = ""

        if "</tool_input>" in _tool_input:
          _tool_input = _tool_input.split("</tool_input>")[0]

        try:
          _tool_input = json.loads(_tool_input.replace("'", '"'))
        except JSONDecodeError:
          logger.error("JSONDecodeError: _tool_input is not a dict: %s", _tool_input)
          try:
            _tool_input = json.loads('{'  + _tool_input.replace("'", '"') + '}')
          except JSONDecodeError:
            logger.error("JSONDecodeError: _tool_input is not a dict: %s", _tool_input)


        if not _tool_input:
          _tool_input = ""

        return AgentAction(tool=_tool, tool_input=_tool_input, log=text)

    elif "<final_answer>" in text:
      long_answer, summary = text.split("<final_answer>")

      if "</final_answer>" in summary:
        summary = summary.split("</final_answer>")[0]

      long_answer = long_answer.replace("<final_answer>", "\n")
      summary = summary.replace("<final_answer>", "\n")

      return AgentFinish(
        return_values={
          "output": f"{long_answer}\n{summary if len(summary) > 50 else ''}"
        }, log=text)
    else:
      return AgentFinish(return_values={"output": text}, log=text)

    return AgentFinish(return_values={"output": "Failed AgentAction. Try again"}, log=text)

  def get_format_instructions(self) -> str:
    raise NotImplementedError

  @property
  def _type(self) -> str:
    return "xml-agent"
