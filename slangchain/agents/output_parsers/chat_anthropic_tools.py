"""chat anthropiuc tools output parsers"""
from typing import Union, List, cast
import logging
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish

logger = logging.getLogger(__name__)

class ChatAnthropicToolsAgentOutputParser(BaseGenerationOutputParser):
  """Chat Anthropic Tools Agent Output Parser"""
  latest_message_only: bool = False
  args_only: bool = False


  class Config:
    """config"""
    extra = "forbid"


  def parse_result(
    self,
    result: List[Generation],
    *,
    partial: bool = False) -> Union[AgentAction, AgentFinish, List[AgentAction], List[AgentFinish]]:
    """Parse a list of candidate model Generations into a specific format.

    Args:
      result: A list of Generations to be parsed. The Generations are assumed
        to be different candidate outputs for a single model input.

    Returns:
      Structured output.
    """
    agent_messages = []

    if not result or not isinstance(result[0], ChatGeneration):
      return None if self.latest_message_only else []


    agent_messages: List = _extract_agent_messages(result[0].message)

    if agent_messages:

      if self.args_only:
        agent_messages = [
          tc.tool_input for tc in agent_messages if isinstance(tc, AgentActionMessageLog)]
      else:
        pass

      if self.latest_message_only:
        return agent_messages[-1] if agent_messages else None

    return agent_messages


def _extract_agent_messages(
  msg: BaseMessage) -> List[Union[AgentActionMessageLog, AgentFinish]]:
  """extract Agent messages"""

  agent_messages = []

  if isinstance(msg.content, list):
    for i, block in enumerate(cast(List[dict], msg.content)):
      if block["type"] == "tool_use":
        tool_id = block["id"]
        tool_name = block["name"]
        tool_input = block["input"]
        content_msg = f"responded: {msg.content}\n" if msg.content else "\n"
        log = (
          f"\nInvoking: `{tool_name}`"
          f" id: `{tool_id}`, index: {i} input: `{tool_input}`"
          f"\n{content_msg}\n")

        agent_messages.append(
          AgentActionMessageLog(
            tool = tool_name,
            tool_input = tool_input,
            log = log,
            message_log=[msg]
          )
        )
      else:
        agent_messages.append(
          AgentFinish(
            return_values={"output": msg.content}, log=str(msg.content)
          )
        )
  else:
    agent_messages.append(
      AgentFinish(
        return_values={"output": msg.content}, log=str(msg.content)
      ))

  return agent_messages
