"""Schemas"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Extra, Field

class ExplainedImage(BaseModel):
  """"ExplainedImage Class"""

  explanation: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

class UrlExplainedImage(ExplainedImage):
  """"UrlExplained Image Class"""

  url: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

  @classmethod
  def from_objs(cls, url: str, explanation: str):
    """from_objs"""
    return cls(
      url = url,
      explanation = explanation
    )

class Base64ImageStringExplainedImage(ExplainedImage):
  """"Base64EncodedExplainedImage Class"""

  source: str
  file_type: Optional[str] = Field(default=None)
  encoded_binary: Optional[str] = Field(default=None)

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

  @classmethod
  def from_objs(
    cls,
    source: str,
    explanation: str,
    file_type: Optional[str] = None,
    encoded_binary: Optional[str] = None):
    """from_objs"""
    return cls(
      source = source,
      file_type = file_type,
      encoded_binary = encoded_binary,
      explanation = explanation
    )

class SeleniumWebElement(BaseModel):
  """Selenium Web Element"""

  element_id: str
  description: str
  element_type: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

  @classmethod
  def from_objs(
    cls,
    element_id: str,
    description: str,
    element_type: str):
    """from_objs"""
    return cls(
      element_id = element_id,
      description = description,
      element_type = element_type
    )


class SeleniumLinkWebElement(SeleniumWebElement):
  """Selenium Link Web Element"""

  url: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

  @classmethod
  def from_objs(
    cls,
    element_id: str,
    description: str,
    element_type: str,
    url: Optional[str] = None):
    """from_objs"""
    return cls(
      element_id = element_id,
      description = description,
      element_type = element_type,
      url = url
    )

class SeleniumActionResult(BaseModel):
  """Selenium Action Result"""

  url: Optional[str] = Field(default="")
  result_status: str
  result: Optional[Dict[str, Any]] = Field(default={})

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = False

  @classmethod
  def from_objs(
    cls,
    url: Optional[str] = None,
    result_status: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None):
    """from_objs"""
    if not url:
      url = ""
    if not result:
      result = {}
    return cls(
      url = url,
      result_status = result_status,
      result = result
    )
