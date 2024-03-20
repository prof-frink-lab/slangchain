"""Url Image Explainer Chain"""
from __future__ import annotations
import sys
import traceback
import re
import logging
from typing import Dict, List, Optional, Any
from io import BytesIO

import base64
import requests
import imghdr

from pydantic import Extra, Field, validator

from slangchain.schemas import UrlExplainedImage, Base64ImageStringExplainedImage

from langchain.schema import Document
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema.messages import BaseMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger()

DEFAULT_PROMPT = "Based on the image, explain in detail"


class UrlImageExplainerChain(Chain):
  """URL image Explainer chain
  """
  url_key: str = "url"
  output_key: str = "result"  #: :meta private:
  prompt: str = DEFAULT_PROMPT

  model_name: Optional[str] = Field(default = "claude-3-haiku-20240307")
  temperature: Optional[float] = Field(default = 0.0)
  max_tokens: Optional[int] = Field(default = 1024)
  explained_images: List[Document] = Field(default = [])

  class Config:
    """Configuration for this pydantic object."""
    extra = Extra.forbid
    arbitrary_types_allowed = True

  @property
  def input_keys(self) -> List[str]:
    """Input keys for chain."""
    return [ self.url_key ]

  @property
  def output_keys(self) -> List[str]:
    """Output keys for chain."""
    return [self.output_key]

  @validator("model_name")
  def validate_model_name(cls, v: str) -> str:
    """Validate that model_name is allowed."""
    if not v.startswith("claude-3") and not v.startswith("gpt-4-vision"):
      raise ValueError(f"model_name {v} not valid.")
    return v

  def _call(
    self,
    inputs: Dict[str, str],
    run_manager: Optional[CallbackManagerForChainRun] = None) -> UrlExplainedImage:
    """Call the internal chain."""
    explained_image: UrlExplainedImage = None
    try:
      url: str = inputs.get(self.url_key)

      if self.model_name.startswith("gpt-4-vision"):
        model = ChatOpenAI(
          model = self.model_name,
          temperature = self.temperature,
          max_tokens = self.max_tokens)
      elif self.model_name.startswith("claude-3"):
        model = ChatAnthropic(
          model = self.model_name,
          temperature = self.temperature,
          max_tokens = self.max_tokens)
      else:
        raise NotImplementedError(f"model not implemented: {self.model_name}")

      image_content = requests.get(url).content
      img_base64 = base64.b64encode(image_content).decode("utf-8")

      file_type = imghdr.what(BytesIO(image_content))

      result: BaseMessage = model.invoke(
        [
          
          HumanMessage(
            content=[
              {
                "type": "text",
                "text": f"{self.prompt}",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/{file_type};base64,{img_base64}",
                }
              }])
        ])
      explained_image = UrlExplainedImage.from_objs(url = url, explanation = result.content)
    except Exception:

      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    return { self.output_key: explained_image }

  @classmethod
  def from_parameters(
    cls,
    model_name: Optional[str] = "claude-3-haiku-20240307",
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = 1024,
    prompt: Optional[str] = DEFAULT_PROMPT,
    metadata: Optional[Dict] = None
    ) -> UrlImageExplainerChain:
    """Instantiate class from splitter and embeddings"""

    if not model_name:
      model_name = "claude-3-haiku-20240307"

    if not prompt:
      prompt = DEFAULT_PROMPT
    else:
      prompt = DEFAULT_PROMPT + f". {prompt}"

    cls(
      model_name = model_name,
      temperature = temperature,
      max_tokens = max_tokens,
      prompt = prompt,
      metadata = metadata
    )

  @property
  def _chain_type(self) -> str:
    return "url_image_explainer_chain"



class Base64ImageStringExplainerChain(Chain):
  """Base64 Image String Image Explainer chain
  """
  base64_image_string_key: str = "base64_image_string"
  source_key: str = "source"
  output_key: str = "result"  #: :meta private:
  prompt: str = DEFAULT_PROMPT

  model_name: Optional[str] = Field(default = "claude-3-haiku-20240307")
  max_tokens: Optional[int] = Field(default = 1024)
  explained_images: List[Document] = Field(default = [])
  temperature: Optional[float] = Field(default = 0.0)

  class Config:
    """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

  @property
  def input_keys(self) -> List[str]:
    """Input keys for chain."""
    return [ self.base64_image_string_key, self.source_key ]

  @property
  def output_keys(self) -> List[str]:
    """Output keys for chain."""
    return [self.output_key]

  @validator("model_name")
  def validate_model_name(cls, v: str) -> str:
    """Validate that model_name is allowed."""
    if not v.startswith("claude-3") and not v.startswith("gpt-4-vision"):
      raise ValueError(f"model_name {v} not valid.")
    return v

  def _call(
    self,
    inputs: Dict[str, str],
    run_manager: Optional[CallbackManagerForChainRun] = None) -> UrlExplainedImage:
    """Call the internal chain."""
    explained_image: UrlExplainedImage = None
    try:
      base64_image_str: str = inputs.get(self.base64_image_string_key)
      source: str = inputs.get(self.source_key)

      if self.model_name.startswith("gpt-4"):
        model = ChatOpenAI(
          model = self.model_name,
          temperature = self.temperature,
          max_tokens = self.max_tokens)
      elif self.model_name.startswith("claude-3"):
        model = ChatAnthropic(
          model = self.model_name,
          temperature = self.temperature,
          max_tokens = self.max_tokens)
      else:
        raise NotImplementedError(f"model not implemented: {self.model_name}")

      file_type = imghdr.what(BytesIO(base64.b64decode(base64_image_str.encode('utf-8'))))

      result: BaseMessage = model.invoke(
        [
          HumanMessage(
            content=[
              {
                "type": "text",
                "text": f"{self.prompt}",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/{file_type};base64,{base64_image_str}"},
              }])
        ])
      explained_image = Base64ImageStringExplainedImage.from_objs(
        source = source, explanation = result.content)
    except Exception:

      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    return { self.output_key: explained_image }

  @classmethod
  def from_parameters(
    cls,
    model_name: Optional[str] = "claude-3-haiku-20240307",
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = 1024,
    prompt: Optional[str] = DEFAULT_PROMPT,
    metadata: Optional[Dict] = None
    ) -> Base64ImageStringExplainerChain:
    """Instantiate class from parameters"""

    if not model_name:
      model_name = "claude-3-haiku-20240307"

    if not prompt:
      prompt = DEFAULT_PROMPT

    cls(
      model_name = model_name,
      max_tokens = max_tokens,
      prompt = prompt,
      temperature = temperature,
      metadata = metadata
    )

  @property
  def _chain_type(self) -> str:
    return "base64_image_string_explainer_chain"
