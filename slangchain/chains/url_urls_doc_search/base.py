"""URL URLs similar document chain"""
from __future__ import annotations
import sys
import traceback
import re
import logging
from typing import Dict, List, Optional

from langchain_core.pydantic_v1 import Extra, Field

from langchain.schema import BaseRetriever
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from langchain.text_splitter import TextSplitter, TokenTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from slangchain.document_loaders.url_urls import UrlUrlsLoader

logger = logging.getLogger()

class UrlUrlsDocSearchChain(Chain):
  """URL similar document chain
  """

  input_key: str = "query"  #: :meta private:
  output_key: str = "result"  #: :meta private:

  embeddings: Embeddings = Field(default_factory = HuggingFaceEmbeddings)
  splitter: TextSplitter = Field(default = None)
  vectorstore: VectorStore = Field(default = None)
  retriever: BaseRetriever = Field(default = None)
  k: int = Field(default=5)
  compressed_search_flag: Optional[bool] = Field(default = False)
  chunk_size: int = Field(default = 1000)
  chunk_overlap: int = Field(default = 200)
  timeout: int = Field(default = 30)

  query_delimiter: str = Field(default = "|")
  queries_delimiter: str = Field(default = "\n")

  similarity_threshold: float = Field(default = 0.9)

  metadata: Optional[Dict] = Field(default = None)

  url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'

  all_docs: List[Document] = Field(default = [])
  relevant_docs: List[Document] = Field(default = [])

  browser_headless_flag: Optional[bool] = Field(default = True)

  class Config:
    """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

  @property
  def input_keys(self) -> List[str]:
    """Input keys for chain."""
    return [self.input_key]

  @property
  def output_keys(self) -> List[str]:
    """Output keys for chain."""
    return [self.output_key]

  def _call(
    self,
    inputs: Dict[str, str],
    run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, str]:
    """Call the internal chain."""
    try:
      # Other keys are assumed to be needed for LLM prediction
      urls = self._get_urls(inputs.get(self.input_key, ""))

      loader = UrlUrlsLoader(
        urls=urls,
        headless=self.browser_headless_flag)
      self.all_docs = loader.load()
      if self.all_docs:
        if not self.splitter:
          self.splitter = TokenTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
          )

        self.all_docs = self.splitter.split_documents(self.all_docs)
        queries = self._get_search_queries(inputs.get(self.input_key, ""))

        if not self.vectorstore:
          self.vectorstore = FAISS.from_documents(self.all_docs, self.embeddings)
        else:
          self.vectorstore.add_documents(self.all_docs)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        relevant_docs: List[Document] = []

        for query in queries:
          relevant_docs += self.retriever.get_relevant_documents(query)

        self.relevant_docs = [
          Document(page_content=doc.page_content, metadata=doc.metadata) \
            for doc in relevant_docs]

    except Exception:

      ex_type, ex_value, ex_traceback = sys.exc_info()
      filename = ex_traceback.tb_frame.f_code.co_filename
      line_number = ex_traceback.tb_lineno
      logger.error(
        "Exception - Type: %s, Exception: %s, Traceback: %s, File: %s, Line: %d", \
        ex_type, ex_value, ex_traceback, filename, line_number)
      traceback.print_exc()

    return { self.output_key: self.relevant_docs }

  @classmethod
  def from_splitter_and_embeddings(
    cls,
    splitter : TextSplitter,
    embeddings: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
    query_delimiter: str,
    queries_delimiter: str,
    k: Optional[int] = 5
    ) -> UrlUrlsDocSearchChain:
    """Instantiate class from splitter and embeddings"""

    cls(
      splitter = splitter,
      embeddings = embeddings,
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap,
      query_delimiter = query_delimiter,
      queries_delimiter = queries_delimiter,
      k = k
    )

  def _get_urls(self, input_prompt: str) -> List[str]:
    urls: List[str] = input_prompt.split(self.queries_delimiter)
    urls = [ url.split(self.query_delimiter)[0].lstrip().rstrip() \
            for url in urls if url]
    urls = [url for url in urls if self._validate_url(url)]
    return urls

  def _validate_url(self, url: str) -> bool:
    if re.match(self.url_pattern, url):
      return True
    return False

  def _get_search_queries(self, input_prompt: str) -> List[str]:
    queries: List[str] = input_prompt.split(self.queries_delimiter)
    queries = [ query.split(self.query_delimiter)[-1].lstrip().rstrip() \
            for query in queries if query]
    queries = list(set(queries))
    return queries

  @property
  def _chain_type(self) -> str:
    return "url_urls_doc_search"
