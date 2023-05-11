"""Task Prioritization chain class"""
from typing import Optional
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.callbacks.manager import Callbacks

class TODOChain(LLMChain):
  """Chain to prioritize tasks."""

  @classmethod
  def from_llm(cls,
    llm: BaseLLM,
    max_task_list_num: int = 5,
    verbose: bool = True,
    callbacks: Optional[Callbacks] = None) -> LLMChain:
    """Get the response parser."""
    todo_template = (
      "You are a planner who is an expert at coming up with a todo list for a given objective."
      " Come up with a todo list for this objective: {objective}"
      f" of no more than {max_task_list_num} tasks."
    )
    prompt = PromptTemplate(
      template=todo_template,
      input_variables=["objective"],
    )
    return cls(prompt=prompt, llm=llm, verbose=verbose, callbacks=callbacks)
  