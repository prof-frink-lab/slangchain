"""Loader that uses unstructured to load HTML files."""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List

from io import BytesIO, StringIO

import requests

from langchain.document_loaders.url_selenium import SeleniumURLLoader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import UnstructuredFileIOLoader

logger = logging.getLogger(__file__)

class UnstructuredURLFileLoader(BaseLoader):
  """Loader that uses unstructured to load HTML files."""

  def __init__(
    self,
    urls: List[str],
    headless: bool=False,
    timeout: int=10,
    max_concurrency=10
  ):
    """Initialize with file path."""

    self.urls = urls
    self.headless = headless
    self.timeout = timeout
    self.max_concurrency = max_concurrency
    self.all_docs: List[Document] = []

  def _process_url(self, url: str, timeout: int=10) -> List[Document]:

    logger.info("Processing url: %s", url)
    content = ""
    docs = []

    response = requests.get(url, timeout=timeout)

    if "text/plain" in response.headers["Content-Type"] or \
      "text/html" in response.headers["Content-Type"]:
      content = response.content.decode("utf-8")

    if "<script" in content and \
      "</script>" in content:
      loader = SeleniumURLLoader([url], headless=self.headless)
      docs = loader.load()

    else:
      if "text/plain" in response.headers["Content-Type"]:
        content = StringIO(response.content.decode("utf-8"))
      else:
        content = BytesIO(response.content)

      loader = UnstructuredFileIOLoader(file=content)
      docs = loader.load()

    for doc in docs:
      doc.metadata = { "source": url }

    return docs

  async def _process_urls_loop(self, urls: List[str], max_concurrency: int) -> List[Document]:

    all_docs: List[Document] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    with ThreadPoolExecutor() as executor:
      loop = asyncio.get_running_loop()

      async def run_task(url: str):
        async with semaphore:
          return await loop.run_in_executor(executor, self._process_url, url)

      tasks = [run_task(url) for url in urls]

      for task in asyncio.as_completed(tasks):
        docs = await task
        all_docs.append(docs)

    all_docs = [item for sublist in all_docs for item in sublist]

    return all_docs

  def load(self) -> List[Document]:
    """Load urls into Documents."""
    self.all_docs = asyncio.run(self._process_urls_loop(self.urls, self.max_concurrency))

    return self.all_docs
