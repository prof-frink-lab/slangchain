"""AWS Schemas"""
from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

META_DATA_TYPE: str = "META_DATA"
LLM_START_TYPE: str = "LLM_START"
LLM_NEW_TOKEN_TYPE: str = "LLM_NEW_TOKEN"
LLM_END_TYPE: str = "LLM_END"
LLM_ERROR_TYPE: str = "LLM_ERROR"
CHAIN_START_TYPE: str = "CHAIN_START"
CHAIN_END_TYPE: str = "CHAIN_END"
CHAIN_ERROR_TYPE: str = "CHAIN_ERROR"
TOOL_START_TYPE: str = "TOOL_START"
TOOL_END_TYPE: str = "TOOL_END"
TOOL_ERROR_TYPE: str = "TOOL_ERROR"
AGENT_ACTION_TYPE: str = "AGENT_ACTION"
TEXT_TYPE: str = "TEXT_TYPE"
AGENT_FINISH_TYPE: str = "AGENT_FINISH"
RESULT_TYPE: str = "RESULT"

PARTITION_KEY_NAME: str = "partition_key"
SORT_KEY_NAME: str = "sort_key"

USER_ID_KEY : str = "user_id"
UNIQUE_ID_KEY : str = "unique_id"
RECORD_TYPE_KEY: str = "record_type"
TIMESTAMP_KEY: str = "timestamp"
PAYLOAD_KEY: str = "payload"
MESSAGE_KEY: str = "message"
RECORD_TYPE_KEY: str = "record_type"
ARGUMENTS_KEY: str = "arguments"
SEQUENCE_NUM_KEY: str = "sequence_number"
S3_BUCKET_KEY: str = "s3_bucket"
TASK_S3_PREFIX_KEY: str = "task_s3_prefix"
TASK_RESULTS_S3_PREFIX_KEY: str = "task_results_s3_prefix"

class DDBPayload(BaseModel):
  """DDB Payload"""
  partition_key : str
  sort_key: str
  record_type: str
  timestamp: float
  sequence_number: int = Field(default=None)
  payload: Dict = Field(default={})

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

class DDBPayloadList(BaseModel):
  """DDB Payloads"""
  __root__: List[DDBPayload]

  def __iter__(self):
    return iter(self.__root__)

  def __getitem__(self, item):
    return self.__root__[item]
