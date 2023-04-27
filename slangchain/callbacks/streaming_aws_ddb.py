"""Callback Handler that streams to DynamoDB."""
import time
import logging

from uuid import uuid4
from typing import Any, Dict, List, Union, Optional
from boto3.resources.factory import ServiceResource

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from slangchain.aws.schemas import (
  LLM_START_TYPE ,
  LLM_NEW_TOKEN_TYPE ,
  LLM_END_TYPE ,
  LLM_ERROR_TYPE ,
  CHAIN_START_TYPE ,
  CHAIN_END_TYPE ,
  CHAIN_ERROR_TYPE ,
  TOOL_START_TYPE ,
  TOOL_END_TYPE ,
  TOOL_ERROR_TYPE ,
  AGENT_ACTION_TYPE ,
  TEXT_TYPE ,
  AGENT_FINISH_TYPE ,
  MESSAGE_KEY,
  ARGUMENTS_KEY,
)
from slangchain.aws.schemas import DDBPayload, DDBPayloadList
from slangchain.aws.dynamodb.base import DDBAppSaver

logger = logging.getLogger()

class StreamingDynamoDBCallbackHandler(BaseCallbackHandler):
  """Callback handler for DynamoDB streaming."""

  def __init__(
    self,
    ddb_resource: ServiceResource = None,
    ddb_table_name: str = None,
    partition_key_value: Optional[str] = None,
    sort_key_prefix_value: Optional[str] = None,
    verbose: Optional[bool] = False
    ) -> None:
    """Initialize callback handler."""

    super().__init__()


    self.ddb_resource = ddb_resource
    self.ddb_table_name = ddb_table_name

    self.partition_key_value = partition_key_value
    self.sort_key_prefix_value = sort_key_prefix_value

    self.uuid = uuid4()

    self.verbose = verbose
    self.sequence_number = 0

  def on_llm_start(
    self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
  ) -> None:
    """Run when LLM starts running."""
    self._log(
      LLM_START_TYPE,
      { MESSAGE_KEY: serialized, "prompts": prompts, ARGUMENTS_KEY: kwargs })

  def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    """Run on new LLM token. Only available when streaming is enabled."""
    self._log(LLM_NEW_TOKEN_TYPE, { MESSAGE_KEY: token, ARGUMENTS_KEY: kwargs })

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    """Run when LLM ends running."""
    self._log(LLM_END_TYPE, { MESSAGE_KEY: dict(response), ARGUMENTS_KEY: kwargs })

  def on_llm_error(
    self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
  ) -> None:
    """Run when LLM errors."""
    self._log(LLM_ERROR_TYPE, { MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_chain_start(
    self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
  ) -> None:
    """Run when chain starts running."""
    self._log(
      CHAIN_START_TYPE,
      { MESSAGE_KEY: serialized, "inputs": inputs, ARGUMENTS_KEY: kwargs })

  def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
    """Run when chain ends running."""
    self._log(CHAIN_END_TYPE, { MESSAGE_KEY: outputs, ARGUMENTS_KEY: kwargs })

  def on_chain_error(
    self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
  ) -> None:
    """Run when chain errors."""
    self._log(CHAIN_ERROR_TYPE, {MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_tool_start(
    self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
  ) -> None:
    """Run when tool starts running."""
    self._log(
      TOOL_START_TYPE,
      { MESSAGE_KEY: serialized, "input_str": input_str, ARGUMENTS_KEY: kwargs })

  def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
    """Run on agent action."""
    self._log(AGENT_ACTION_TYPE, {  MESSAGE_KEY: action._asdict(), ARGUMENTS_KEY: kwargs })

  def on_tool_end(self, output: str, **kwargs: Any) -> None:
    """Run when tool ends running."""
    self._log(TOOL_END_TYPE, {MESSAGE_KEY: output, ARGUMENTS_KEY: kwargs})

  def on_tool_error(
    self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
  ) -> None:
    """Run when tool errors."""
    self._log(TOOL_ERROR_TYPE, { MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_text(self, text: str, **kwargs: Any) -> None:
    """Run on arbitrary text."""
    self._log(TEXT_TYPE, {MESSAGE_KEY: text, ARGUMENTS_KEY: kwargs })

  def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
    """Run on agent end."""
    self._log(AGENT_FINISH_TYPE, { MESSAGE_KEY: finish._asdict(), ARGUMENTS_KEY: kwargs })

  def _log(self, record_type:str, payload:Dict):
    """save item to ddb"""
    resp = {}
    timestamp = time.time()

    self.sequence_number += 1

    sort_key = self.sort_key_prefix_value \
      + f"#{self.uuid}" \
      + f"#{timestamp}" \
      + f"#{record_type}" \
      + f"#{self.sequence_number:06d}"

    payload = DDBPayload(
      partition_key=self.partition_key_value,
      sort_key=sort_key,
      record_type=record_type,
      timestamp=timestamp,
      sequence_number=self.sequence_number,
      payload=payload)

    payloads = DDBPayloadList.parse_obj([payload])
    if self.ddb_resource and self.ddb_table_name and payloads:
      logger.debug("Saving to DynamoDB %s", payloads)
      saver = DDBAppSaver.from_dynamodb_resource_and_table_name(
        self.ddb_resource, self.ddb_table_name)
      saver.save(payloads)
    return resp
