"""URL similar document chain"""
from __future__ import annotations
import sys
import traceback
import re
import logging
from typing import Dict, List

from pydantic import Extra, Field

from ruminat_langchain.document_loaders.url_file import UnstructuredURLFileLoader

from langchain.schema import BaseRetriever
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

logger = logging.getLogger()

class UrlCompressedDocSearchChain(Chain):
  """URL similar document chain
  """

  input_key: str = "query"  #: :meta private:
  output_key: str = "result"  #: :meta private:

  embeddings: Embeddings = Field(default_factory = HuggingFaceEmbeddings)
  splitter: TextSplitter = Field(default = None)
  retriever: BaseRetriever = Field(default = None)
  chunk_size: int = Field(default = 1000)
  chunk_overlap: int = Field(default = 200)
  timeout: int = Field(default = 30)

  query_delimiter: str = Field(default = "|")
  queries_delimiter: str = Field(default = "\n")

  similarity_threshold: float = Field(default = 0.9)

  url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'

  all_docs: List[Document] = Field(default = [])
  relevant_docs: List[Document] = Field(default = [])

  browser_headless_flag: bool = Field(default = True)

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

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    """Call the internal chain."""
    try:
      # Other keys are assumed to be needed for LLM prediction
      urls = self._get_urls(inputs.get(self.input_key, ""))
      loader = UnstructuredURLFileLoader(urls, headless=self.browser_headless_flag)
      self.all_docs = loader.load()

      if not self.splitter:
        self.splitter = RecursiveCharacterTextSplitter(
          chunk_size = self.chunk_size,
          chunk_overlap = self.chunk_overlap
        )

      self.all_docs = self.splitter.split_documents(self.all_docs)

      queries = self._get_search_queries(inputs.get(self.input_key, ""))

      retriever = FAISS.from_documents(self.all_docs, self.embeddings).as_retriever()
      redundant_filter = EmbeddingsRedundantFilter(
        embeddings=self.embeddings)
      relevant_filter = EmbeddingsFilter(
        embeddings=self.embeddings, similarity_threshold=self.similarity_threshold)
      pipeline_compressor = DocumentCompressorPipeline(
        transformers=[self.splitter, redundant_filter, relevant_filter])
      self.retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever)

      for query in queries:
        self.relevant_docs += self.retriever.get_relevant_documents(query)

      self.relevant_docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata) \
          for doc in self.relevant_docs]

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
    similarity_threshold: float,
    ) -> UrlCompressedDocSearchChain:
    """Instantiate class from splitter and embeddings"""

    cls(
      splitter = splitter,
      embeddings = embeddings,
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap,
      query_delimiter = query_delimiter,
      queries_delimiter = queries_delimiter,
      similarity_threshold = similarity_threshold
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
    return "url_compressed_doc_search"
