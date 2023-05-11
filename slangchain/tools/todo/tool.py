"""TODO Creator Tool"""
from typing import Optional

from langchain.tools.base import BaseTool
from langchain.llms import BaseLLM
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from slangchain.chains.todo.base import TODOChain

class TODOCreatorTool(BaseTool):
  """Tool that has capability to request web content 
  and return the text that is most similar to the query."""

  max_task_list_num: int = 5
  llm: BaseLLM

  name = "TODO creator tool"
  description = """
    A tool that creates a todo list based on your objective.
    Input: an objective to create a todo list for. Output: a todo list for that objective.
    lease be very clear what the objective is!"""

  def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
    """Use the tool."""
    chain = TODOChain.from_llm(llm=self.llm, max_task_list_num=self.max_task_list_num)
    return chain.run(query)

  async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("UrlCompressedDocSearch does not support async")
