"""Url Image Explainer"""
from uuid import uuid4
import json
from typing import Optional, Dict, List, Type, Any
from pydantic import Field, BaseModel

from slangchain.schemas import UrlExplainedImage
from slangchain.chains.image_explainer.base import DEFAULT_PROMPT, UrlImageExplainerChain

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools.base import BaseTool

NAME = "Web_Image_Explainer"
DESCRIPTION =  (
  "A tool that describes and explains a web image in detail.")

class UrlImageExplainerInput(BaseModel):
  """Input for UrlImageUrlsDocLoader."""
  url: str = Field(
    ...,
    description=(
      " Input must be an image url that starts with http:// or https://"
    ))
  query: Optional[str] = Field(
    default=None,
    description=(
      " Optional input query, to query, interrogate or ask questions of the image content."
    ))


class UrlImageExplainer(BaseTool):
  """Url Image Explainer"""

  args_schema: Type[BaseModel] = UrlImageExplainerInput

  model_name: Optional[str] = Field(default = "claude-3-haiku-20240307")

  max_tokens: Optional[int] = Field(default = 1024)

  metadata: Optional[Dict] = Field(default = None)

  name: str = "Web_Image_Explainer"
  description: str

  output_bucket_name: Optional[str] = Field(default=None)
  s3_output_prefix: Optional[str] = Field(default=None)

  @classmethod
  def from_parameters(
    cls,
    name: Optional[str] = NAME,
    description: Optional[str] = DESCRIPTION,
    max_tokens: Optional[int] = 1024,
    model_name: Optional[str] = "claude-3-haiku-20240307",
    output_bucket_name: Optional[str] = None,
    s3_output_prefix: Optional[str] = None,
    callbacks: List[BaseCallbackHandler] = None,
    metadata: Optional[Dict] =  None,
    ):
    """from parameters"""

    if not name:
      name = NAME

    if not description:
      description = DESCRIPTION

    if not model_name:
      model_name = "claude-3-haiku-20240307"

    return cls(
      name = name,
      description = description,
      model_name = model_name,
      max_tokens = max_tokens,
      output_bucket_name = output_bucket_name,
      s3_output_prefix = s3_output_prefix,
      metadata = metadata,
      callbacks = callbacks
    )


  def _run(
    self,
    url: str,
    query: Optional[str] = None,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""

    results: Dict[str, Any] = {}

    chain_args = {
      "model_name": self.model_name,
      "max_tokens": self.max_tokens
    }

    if query:
      chain_args["prompt"] = query

    chain = UrlImageExplainerChain(**chain_args)
    explainer_result: UrlExplainedImage = chain.run(url = url)

    if explainer_result:
      results = explainer_result.dict()

    return json.dumps(results)

  async def _arun(
    self,
    url: str,
    query: Optional[str] = None,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("UrlImageExplainer does not support async")
