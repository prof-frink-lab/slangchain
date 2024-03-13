"""Google Serper search"""
from typing import Optional
from langchain_core.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

class GoogleResultsSearch(BaseTool):
  """Tool that has capability to query the Google Search API and get back json."""

  name = "Google Search Results JSON"
  description = (
      "A wrapper around Google Web Search. "
      "Useful for when you need to answer questions about current events. "
      "Input should be a search query. Output is a JSON array of the query results"
      " Output is a json array with 'title', 'link', 'snippet', 'position' keys")

  search_type: str = "search"
  results_key: str = "organic"
  num_results: int = 5

  api_wrapper: GoogleSerperAPIWrapper = Field(
    default_factory=GoogleSerperAPIWrapper
  )

  def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""
    self.api_wrapper = GoogleSerperAPIWrapper(type = self.search_type, k = self.num_results)
    results = self.api_wrapper.results(query)
    results = results.get(self.results_key, [])
    results = results[: self.num_results]
    return str(results)

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("BingSearchResults does not support async")

class GoogleNewsResultsSearch(BaseTool):
  """Tool that has capability to query the Google Search API and get back json."""

  name = "Google News Search Results JSON"
  description = (
      "A wrapper around Google News Search. "
      "Useful for when you need to answer questions about current events based on news articles. "
      "Input should be a search query. Output is a JSON array of the query results"
      " Output is a json array with title', 'link', 'snippet', 'date',"
      " 'source', 'imageUrl', 'position' keys")

  search_type: str = "news"
  results_key: str = "news"
  num_results: int = 5

  api_wrapper: GoogleSerperAPIWrapper = Field(
    default_factory=GoogleSerperAPIWrapper
  )

  def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""
    self.api_wrapper = GoogleSerperAPIWrapper(type = self.search_type, k = self.num_results)
    results = self.api_wrapper.results(query)
    results = results.get(self.results_key, [])
    results = results[: self.num_results]
    return str(results)

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("BingSearchResults does not support async")

class GooglePlacesResultsSearch(BaseTool):
  """Tool that has capability to query the Google Search API and get back json."""

  name = "Google Places Search Results JSON"
  description = (
      "A wrapper around Google Places Search. "
      "Useful for when you need to answer questions about current events based on their location. "
      "Input should be a search query. Output is a JSON array of the query results."
      " Output is a json array with 'position', 'title', 'address', 'latitude'"
      " , 'longitude', 'rating' , 'ratingCount', 'priceLevel', 'locationHint',"
      " 'category', 'phoneNumber', 'website', 'cid', 'placeId' keys")

  search_type: str = "places"
  results_key: str = "places"
  num_results: int = 5

  api_wrapper: GoogleSerperAPIWrapper = Field(
    default_factory=GoogleSerperAPIWrapper
  )

  def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""
    self.api_wrapper = GoogleSerperAPIWrapper(type = self.search_type, k = self.num_results)
    results = self.api_wrapper.results(query)
    results = results.get(self.results_key, [])
    results = results[: self.num_results]
    return str(results)

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("BingSearchResults does not support async")
