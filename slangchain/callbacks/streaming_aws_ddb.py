"""Callback Handler that streams to DynamoDB."""
import time
import logging

from uuid import uuid4, UUID
from typing import Any, Dict, List, Union, Optional
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

def import_boto3() -> Any:
  """boto3 importer function"""
  try:
    import boto3  # noqa: F401
  except ImportError:
    raise ImportError(
      "To use the boto3 callback manager you need to have the "
      "`boto3` python package installed. Please install it with"
      " `pip install boto3`"
    )
  return boto3

def _get_ddb_table(table_name:str) -> Any:
  _boto3 = import_boto3()
  resource = _boto3.resource("dynamodb")
  ddb_table = resource.Table(table_name)
  return ddb_table

def _save_payload(
    table_resource: Any,
    partition_key_value: str,
    sort_key_value: str,
    record_type: str,
    timestamp: float,
    sequence_number: int,
    payload: Dict[str, Any]) -> None:

  ddb_payload = DDBPayload(
      partition_key=partition_key_value,
      sort_key=sort_key_value,
      record_type=record_type,
      timestamp=timestamp,
      sequence_number=sequence_number,
      payload=payload)


  ddb_payloads = DDBPayloadList.parse_obj([ddb_payload])
  if table_resource and ddb_payloads:
    logger.debug("Saving to DynamoDB %s", ddb_payloads)
    saver = DDBAppSaver.from_table_resource(table_resource, ddb_payloads)
    saver()

class StreamingDynamoDBCallbackHandler(BaseCallbackHandler):
  """Callback handler for DynamoDB streaming."""

  def __init__(
    self,
    ddb_table_name: str = None,
    partition_key_value: Optional[str] = None,
    sort_key_prefix_value: Optional[str] = None,
    verbose: Optional[bool] = False
    ) -> None:
    """Initialize callback handler."""

    super().__init__()

    self.ddb_table_name = ddb_table_name
    self.table_resource = None

    self.partition_key_value = partition_key_value
    self.sort_key_prefix_value = sort_key_prefix_value

    self.uuid = uuid4()

    self.verbose = verbose
    self.sequence_number = 0

  def on_llm_start(
    self,
    serialized: Dict[str, Any],
    prompts: List[str],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when LLM starts running."""
    self._log(
      LLM_START_TYPE,
      { MESSAGE_KEY: serialized, "prompts": prompts, ARGUMENTS_KEY: kwargs })

  def on_llm_new_token(
    self,
    token: str,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any) -> None:
    """Run on new LLM token. Only available when streaming is enabled."""
    self._log(LLM_NEW_TOKEN_TYPE, { MESSAGE_KEY: token, ARGUMENTS_KEY: kwargs })

  def on_llm_end(
    self,
    response: LLMResult,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when LLM ends running."""
    self._log(LLM_END_TYPE, { MESSAGE_KEY: dict(response), ARGUMENTS_KEY: kwargs })

  def on_llm_error(
    self,
    error: Union[Exception, KeyboardInterrupt],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when LLM errors."""
    self._log(LLM_ERROR_TYPE, { MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_chain_start(
    self,
    serialized: Dict[str, Any],
    inputs: Dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when chain starts running."""
    self._log(
      CHAIN_START_TYPE,
      { MESSAGE_KEY: serialized, "inputs": inputs, ARGUMENTS_KEY: kwargs })

  def on_chain_end(
    self,
    outputs: Dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when chain ends running."""
    self._log(CHAIN_END_TYPE, { MESSAGE_KEY: outputs, ARGUMENTS_KEY: kwargs })

  def on_chain_error(
    self,
    error: Union[Exception, KeyboardInterrupt],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when chain errors."""
    self._log(CHAIN_ERROR_TYPE, {MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_tool_start(
    self,
    serialized: Dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when tool starts running."""
    self._log(
      TOOL_START_TYPE,
      { MESSAGE_KEY: serialized, "input_str": input_str, ARGUMENTS_KEY: kwargs })

  def on_tool_end(
    self,
    output: str,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when tool ends running."""
    self._log(TOOL_END_TYPE, {MESSAGE_KEY: output, ARGUMENTS_KEY: kwargs})

  def on_tool_error(
    self,
    error: Union[Exception, KeyboardInterrupt],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run when tool errors."""
    self._log(TOOL_ERROR_TYPE, { MESSAGE_KEY: str(error), ARGUMENTS_KEY: kwargs })

  def on_text(
    self,
    text: str,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run on arbitrary text."""
    self._log(TEXT_TYPE, {MESSAGE_KEY: text, ARGUMENTS_KEY: kwargs })

  def on_agent_action(
    self,
    action: AgentAction,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> Any:
    """Run on agent action."""
    self._log(AGENT_ACTION_TYPE, {  MESSAGE_KEY: action._asdict(), ARGUMENTS_KEY: kwargs })

  def on_agent_finish(
    self,
    finish: AgentFinish,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **kwargs: Any,
  ) -> None:
    """Run on agent end."""
    self._log(AGENT_FINISH_TYPE, { MESSAGE_KEY: finish._asdict(), ARGUMENTS_KEY: kwargs })

  def _log(self, record_type:str, payload:Dict):
    """save item to ddb"""

    resp = {}
    timestamp = time.time()

    self.table_resource = _get_ddb_table(self.ddb_table_name)

    self.sequence_number += 1

    sort_key_value = self.sort_key_prefix_value \
      + f"#{self.uuid}" \
      + f"#{timestamp}" \
      + f"#{record_type}" \
      + f"#{self.sequence_number:06d}"

    _save_payload(
      table_resource=self.table_resource,
      partition_key_value=self.partition_key_value,
      sort_key_value=sort_key_value,
      record_type=record_type,
      timestamp=timestamp,
      sequence_number=self.sequence_number,
      payload=payload)

    return resp
