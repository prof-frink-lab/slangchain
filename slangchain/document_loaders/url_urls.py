"""Loader that loads HREF urls from a parent URL."""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, List, Any, Optional
from urllib.request import Request, urlopen

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

import requests

from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)

class UrlUrlsLoader(BaseLoader):
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

  def _get_bs_urls(self, url: str, metadata: Optional[Dict[str, Any]]=None) -> List[Document]:
    links: List[str] = []
    html_page: bytes = None

    req = Request(url)
    with urlopen(req) as response:
      html_page = response

    soup = BeautifulSoup(html_page, "lxml")

    for link in soup.findAll('a'):
      links.append(
        Document(
          page_content=f'"link_text": {link.text}, "href": {link.get("href")}',
          metadata=metadata))
    print(f"links: {links}")
    return links

  def _get_selenium_urls(self, url: str, metadata: Optional[Dict[str, Any]]=None) -> List[Document]:
    links: List[str] = []
    driver : Chrome = None

    chrome_options = ChromeOptions()


    if self.headless:
      chrome_options.add_argument("--headless")
    params = {
      "options": chrome_options
    }

    driver = Chrome(**params)
    driver.get(url=url)
    links = driver.find_elements(by=By.TAG_NAME, value="a")
    links = [
        Document(
          page_content= f'"link_text": {link.text}, "href": {link.get_attribute("href")}',
          metadata=metadata) \
      for link in links if link.get_attribute("href") and \
        link.get_attribute("href") !="javascript:void(0);" ]

    return links

  def _process_url(self, url: str, timeout: int=10) -> List[Document]:

    content = ""
    docs = []
    metadata = { "source": url }
    response = requests.get(url, timeout=timeout)

    if "text/plain" in response.headers["Content-Type"] or \
      "text/html" in response.headers["Content-Type"]:
      content = response.content.decode("utf-8")

    if "<script" in content and \
      "</script>" in content:
      docs = self._get_selenium_urls(url=url, metadata=metadata)
    else:
      docs = self._get_bs_urls(url=url, metadata=metadata)

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
    self.all_docs = []
    logger.debug("urls: %s", self.urls)
    self.all_docs = asyncio.run(
      self._process_urls_loop(self.urls, self.max_concurrency))

    return self.all_docs
