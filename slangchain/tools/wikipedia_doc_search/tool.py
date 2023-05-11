"""URL Compressed Search tool"""
import json
from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from slangchain.chains.wikipedia_doc_search.base import WikipediaDocSearchChain


class WikipediaDocSearch(BaseTool):
  """Tool that has capability to request web content 
  and return the text that is most similar to the query."""

  query_delimiter: str = "|"
  queries_delimiter: str = "\n"
  load_max_docs: int = Field(default = 5)

  chunk_size: int = Field(default = 1000)
  chunk_overlap: int = Field(default = 200)
  embeddings: Embeddings = Field(default = None)
  text_splitter: TextSplitter = Field(default = None)
  similarity_threshold: float = Field(default = 0.7)

  input_key: str = Field(default = "query")
  output_key: str = Field(default = "result")

  name = "Wikipedia"
  description = """A wrapper around Wikipedia.
    Useful for when you need to answer general questions about
    people, places, companies, facts, historical events, or other subjects.
    Input should be a search query."""

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

    chain = WikipediaDocSearchChain(
      load_max_docs = self.load_max_docs,
      splitter = text_splitter,
      embeddings = embeddings,
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap,
      similarity_threshold = self.similarity_threshold
    )

    result = chain.run({self.input_key: query})

    return json.dumps([ doc.dict() for doc in result ])

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("WikipediaDocSearch does not support async")
