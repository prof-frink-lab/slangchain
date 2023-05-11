"""URL Compressed Search tool"""
import json
from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from slangchain.chains.url_compressed_doc_search.base import UrlCompressedDocSearchChain

class UrlCompressedDocSearch(BaseTool):
  """Tool that has capability to request web content 
  and return the text that is most similar to the query."""

  query_delimiter: str = "|"
  queries_delimiter: str = "\n"
  num_results: int = Field(default = 5)

  chunk_size: int = Field(default = 1000)
  chunk_overlap: int = Field(default = 200)
  embeddings: Embeddings = Field(default = None)
  text_splitter: TextSplitter = Field(default = None)
  similarity_threshold: float = Field(default = 0.7)

  input_key: str = Field(default = "query")
  output_key: str = Field(default = "result")

  local_cache_path: str = Field(default = "")

  browser_headless_flag: Optional[bool] = Field(default = True)

  name = "Web URL Content Search"
  description =  f"""
    A portal to the internet. Use this when you need to get specific content
    from any of the web content types (original media type of the resource).
    The tool will return the content most similar to a search query.
    Input must be a url (i.e. https://www.bbc.com/news) and a search query
    (i.e. Find articles on the solar eclipse) delimited by {str(query_delimiter)}
    (i.e. https://www.bbc.com/news{str(query_delimiter)}Find articles on the solar eclipse).
    There can be mulitple inputs, delimited by {str(queries_delimiter)}
    (i.e. https://www.bbc.com/news{str(query_delimiter)}Find articles on the solar eclipse{str(queries_delimiter)}
          https://edition.cnn.com/{str(query_delimiter)}What is the breaking news?{str(queries_delimiter)}').
    The output will a json response where the page_content key.
    It contains the content and meta_data key contains the url source. 
    Constraint: 
    1. Only use urls from previous Observations or results returned from your tools.
    2. Only use search queries from a previous Action Inputs.
    3. Only search a url if you are certain it'll help you answer your question. """


  def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap
    ) if not self.text_splitter else self.text_splitter

    embeddings = HuggingFaceEmbeddings() if not self.embeddings else self.embeddings

    chain = UrlCompressedDocSearchChain(
      splitter = text_splitter,
      embeddings = embeddings,
      browser_headless_flag = self.browser_headless_flag,
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap,
      query_delimiter = self.query_delimiter,
      queries_delimiter = self.queries_delimiter,
      similarity_threshold = self.similarity_threshold
    )

    result = chain.run({self.input_key: query})

    return json.dumps([ doc.dict() for doc in result ])

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("UrlCompressedDocSearch does not support async")
