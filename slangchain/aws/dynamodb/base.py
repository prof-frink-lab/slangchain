"""AWS DynamoDB classes"""
import sys
import logging
import traceback
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any

from slangchain.aws.schemas import (
  PARTITION_KEY_NAME,
  SORT_KEY_NAME,
  DDBPayloadList)
from langchain_core.pydantic_v1 import BaseModel, Field, Extra, root_validator


logger = logging.getLogger(__name__)

class DDBAppLoader(BaseModel):
  """AWS Secrets Environment Loader"""

  table_resource: Any
  partition_key_value: str
  sort_key_value: Optional[str] = Field(default=None)

  ddb_payloads : Optional[DDBPayloadList] = Field(default=[])

  class Config:
    """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    """Validate that api key and python package exists in environment."""
    try:
      import boto3

    except ImportError:
      raise ValueError(
        "Could not import boto3 python package. "
        "Please install it with `pip install boto3`."
      )
    return values

  def __call__(self) -> DDBPayloadList:
    try:
      self.ddb_payloads = self._get_items_from_ddb()
    except Exception:
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    return self.ddb_payloads


  def load(self) -> DDBPayloadList:
    """load payloads"""
    return self()

  @property
  def payloads(self) -> DDBPayloadList:
    """payloads getter function"""
    return self.ddb_payloads

  @classmethod
  def from_table_resource(
    cls,
    table_resource: Any,
    partition_key_value: str,
    sort_key_value: str) -> None:
    """load DDB records from table name"""

    return cls(
      table_resource=table_resource,
      partition_key_value=partition_key_value,
      sort_key_value=sort_key_value
    )

  def _get_items_from_ddb(self) -> DDBPayloadList:
    """Get items from ddb"""
    items = []
    condition = f"{PARTITION_KEY_NAME} = :partition_key_value"
    expression = {
      ':partition_key_value': self.partition_key_value
    }
    if self.sort_key_value:
      condition += f" AND begins_with({SORT_KEY_NAME}, :sort_key_value )"
      expression[':sort_key_value'] = self.sort_key_value

    response = self.table_resource.query(
      KeyConditionExpression= condition,
      ExpressionAttributeValues=expression
    )
    items = response["Items"]
    if items:
      items = self._convert_decimals_to_numbers(items)
      items = DDBPayloadList.parse_obj(items)
    return items

  def _convert_decimals_to_numbers(
    self, input_obj:Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """Replace deimals in dict"""

    if isinstance(input_obj, list):
      for i, _ in enumerate(input_obj):
        input_obj[i] = self._convert_decimals_to_numbers(input_obj[i])
      return input_obj

    if isinstance(input_obj, dict):
      for k in input_obj.keys():
        input_obj[k] = self._convert_decimals_to_numbers(input_obj[k])
      return input_obj

    if isinstance(input_obj, Decimal):
      return self._convert_decimal_to_number(input_obj)

    return input_obj

  def _convert_decimal_to_number(self, number: Decimal) -> Union[float, int]:
    """convert Decimal to float or int"""

    if float(number) % 1 == 0:
      return int(number)

    return float(number)


class DDBAppSaver(BaseModel):
  """AWS Secrets Environment Loader"""

  table_resource: Any
  ddb_payloads: DDBPayloadList

  class Config:
    """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    """Validate that api key and python package exists in environment."""
    try:
      import boto3

    except ImportError:
      raise ValueError(
        "Could not import boto3 python package. "
        "Please install it with `pip install boto3`."
      )
    return values

  def __call__(self) -> None:
    try:
      payloads = self.ddb_payloads.dict().get("__root__", [])
      payloads = [self._convert_number_to_decimals(i) for i in payloads]
      with self.table_resource.batch_writer() as batch:
        for payload in payloads:
          batch.put_item(Item=payload)
    except Exception:
      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

  def save(self, ddb_payloads: DDBPayloadList):
    """save payloads"""
    self.ddb_payloads = ddb_payloads
    self()

  @classmethod
  def from_table_resource(
    cls,
    table_resource: Any,
    ddb_payloads: DDBPayloadList) -> None:
    """load DDB records from table name"""

    return cls(
      table_resource=table_resource,
      ddb_payloads=ddb_payloads
    )

  def _convert_number_to_decimals(
    self,
    input_obj:Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """Replace with decimals in dict"""
    if isinstance(input_obj, list):
      for i, _ in enumerate(input_obj):
        input_obj[i] = self._convert_number_to_decimals(input_obj[i])
      return input_obj

    if isinstance(input_obj, dict):
      for k in input_obj.keys():
        input_obj[k] = self._convert_number_to_decimals(input_obj[k])
      return input_obj

    if isinstance(input_obj, (int, float)):
      return Decimal(str(input_obj))

    return input_obj
