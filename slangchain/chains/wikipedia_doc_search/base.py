"""Wikipedia similar document chain"""
from __future__ import annotations
import sys
import traceback
import logging
from typing import Dict, List, Optional

from pydantic import Extra, Field

from langchain.schema import BaseRetriever
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

logger = logging.getLogger()

class WikipediaDocSearchChain(Chain):
  """Wikipedia similar document chain
  """

  input_key: str = "query"  #: :meta private:
  output_key: str = "result"  #: :meta private:
  load_max_docs: int = Field(default=5)

  embeddings: Embeddings = Field(default_factory = HuggingFaceEmbeddings)
  splitter: TextSplitter = Field(default = None)
  retriever: BaseRetriever = Field(default = None)
  chunk_size: int = Field(default = 1000)
  chunk_overlap: int = Field(default = 200)
  timeout: int = Field(default = 30)

  similarity_threshold: float = Field(default = 0.9)


  all_docs: List[Document] = Field(default = [])
  relevant_docs: List[Document] = Field(default = [])

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
      query = inputs.get(self.input_key, "")
      loader = WikipediaLoader(query=query, load_max_docs=self.load_max_docs)
      self.all_docs = loader.load()

      if not self.splitter:
        self.splitter = RecursiveCharacterTextSplitter(
          chunk_size = self.chunk_size,
          chunk_overlap = self.chunk_overlap
        )

      self.all_docs = self.splitter.split_documents(self.all_docs)

      retriever = FAISS.from_documents(self.all_docs, self.embeddings).as_retriever()
      redundant_filter = EmbeddingsRedundantFilter(
        embeddings=self.embeddings)
      relevant_filter = EmbeddingsFilter(
        embeddings=self.embeddings, similarity_threshold=self.similarity_threshold)
      pipeline_compressor = DocumentCompressorPipeline(
        transformers=[self.splitter, redundant_filter, relevant_filter])
      self.retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever)

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
    load_max_docs : int,
    splitter : TextSplitter,
    embeddings: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
    query_delimiter: str,
    queries_delimiter: str,
    similarity_threshold: float,
    ) -> WikipediaDocSearchChain:
    """Instantiate class from splitter and embeddings"""

    cls(
      load_max_docs = load_max_docs,
      splitter = splitter,
      embeddings = embeddings,
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap,
      query_delimiter = query_delimiter,
      queries_delimiter = queries_delimiter,
      similarity_threshold = similarity_threshold
    )

  @property
  def _chain_type(self) -> str:
    return "wikipedia_doc_search"
