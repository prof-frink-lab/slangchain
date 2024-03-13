"""SlangChain Schema"""
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Extra

class TaskResult(BaseModel):
  """Task Result Payload"""
  task_id : str
  task_name: str
  result: str

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True

class TaskResultList(BaseModel):
  """DDB Payloads"""
  __root__: List[TaskResult]

  def __iter__(self):
    return iter(self.__root__)

  def __getitem__(self, item):
    return self.__root__[item]

  @property
  def task_results(self) -> List[str]:
    """get task results string"""
    task_results : List[str] = []

    for task_result in self.__root__:
      task_results.append(
        f"{task_result.task_name} \n\n {task_result.result}\n\n\n")

    return task_results
