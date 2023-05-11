"""BabyAGI agent."""
from collections import deque
from typing import Any, Dict, List, Optional
import logging
import faiss

from pydantic import BaseModel, Field

from langchain.experimental.autonomous_agents.baby_agi import TaskCreationChain
from langchain.experimental.autonomous_agents.baby_agi import TaskPrioritizationChain

from langchain.schema import Document
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks, CallbackManagerForChainRun
from langchain.chains.base import Chain

from langchain.vectorstores.base import VectorStore
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS

from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.agents import AgentExecutor

from slangchain.autonomous_agents.schemas import TaskResult, TaskResultList

logger = logging.getLogger(__name__)


def _create_default_task_embeddings() -> Embeddings:
  """Create default task embeddings"""
  embeddings_model = HuggingFaceEmbeddings()
  return embeddings_model

def _create_default_vectorstore(embeddings: Embeddings) -> VectorStore:
  """Create default vector store"""
  embedding_size = len(embeddings.embed_query("test"))
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore = FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})
  return vectorstore

class PersistentBabyAGI(Chain, BaseModel):
  """Controller model for the PersistentBabyAGI agent."""

  task_list: deque = Field(default_factory=deque)
  task_creation_chain: Chain = Field(...)
  task_prioritization_chain: Chain = Field(...)
  execution_chain: Chain = Field(...)
  task_id_counter: int = Field(1)

  task_id_counter: int = Field(1)
  task_embeddings: Embeddings = Field(init=False)
  task_vectorstore: VectorStore = Field(init=False)
  task_result_vectorstore: VectorStore = Field(init=False)
  task_result_retriever: TimeWeightedVectorStoreRetriever = Field(init=False)
  max_iterations: Optional[int] = Field(default=None)
  max_task_list_num: Optional[int] = Field(default=None)
  max_iterations: Optional[int] = None
  results : TaskResultList = None

  class Config:
    """Configuration for this pydantic object."""

    arbitrary_types_allowed = True

  def add_task(self, task: Dict) -> None:
    """add task function"""
    self.task_list.append(task)

  def print_task_list(self):
    """Print task list function"""
    logger.info("*****TASK LIST*****")
    for task in self.task_list:
      logger.info("%s: %s", task['task_id'], task['task_name'])

  def print_next_task(self, task: Dict):
    """Print next task function"""
    logger.info("*****NEXT TASK*****")
    logger.info("%s: %s", task['task_id'], task['task_name'])

  def print_task_result(self, result: str):
    """Print task result function"""
    logger.info("*****RESULT*****")
    logger.info("%s", result)

  @property
  def input_keys(self) -> List[str]:
    return ["objective"]

  @property
  def output_keys(self) -> List[str]:
    return []

  def get_next_task(
    self,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)

    response = self.task_creation_chain.run(
      result=result,
      task_description=task_description,
      incomplete_tasks=incomplete_tasks,
      objective=objective)

    new_tasks = response.split('\n')

    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

  def prioritize_tasks(
    self,
    this_task_id: int,
    task_list: List[Dict],
    objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = self.task_prioritization_chain.run(
      task_names=task_names, next_task_id=next_task_id, objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []

    for task_string in new_tasks:
      if not task_string.strip():
        continue
      task_parts = task_string.strip().split(".", 1)

      if len(task_parts) == 2:
        task_id = task_parts[0].strip()
        task_name = task_parts[1].strip()
        prioritized_task_list.append({"task_id": task_id, "task_name": task_name})

    return prioritized_task_list


  def _get_top_tasks(self, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = self.task_result_vectorstore.similarity_search_with_score(query, k=k)
    if not results:
      return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task_name']) for item in sorted_results]

  def _get_tasks_from_vectorstore(self) -> List[Dict[str, str]]:
    task_list: List[Dict[str, str]] = []
    if self.task_vectorstore:
      index = self.task_vectorstore.index
      num_vectors = index.ntotal
      for i in range(num_vectors):
        vector = index.reconstruct(i)
        doc = self.task_vectorstore.similarity_search_by_vector(vector, k=1)
        task_list.append(doc[0].metadata)
    return task_list

  def execute_task(
    self,
    objective: str,
    task: str,
    k: int = 5) -> str:
    """Execute a task."""
    context = self._get_top_tasks(query=objective, k=k)
    return self.execution_chain.run(objective=objective, context=context, task=task)


  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:
    """Run the agent."""
    objective = inputs['objective']

    first_task = inputs.get(
      "first_task", "Make a todo list")
    self.task_list = deque(self._get_tasks_from_vectorstore())

    if not self.task_list:
      self.add_task({"task_id": 1, "task_name": first_task})

    num_iters = 0

    while True:
      if self.task_list:
        self.print_task_list()

        # Step 1: Pull the first task
        task = self.task_list.popleft()
        self.print_next_task(task)
        this_task_id = None
        try:
          this_task_id = int(task.get("task_id"))
        except ValueError:
          logger.error( \
            "Unable to convert task id %s to a number. Skipping over.", task.get("task_id"))
        if not this_task_id:
          continue

        # Step 2: Execute the task
        result = self.execute_task(
          objective, task["task_name"]
        )

        self.print_task_result(result)

        task_result : TaskResult = TaskResult(
              task_id=num_iters,
              task_name=f"{task['task_id']}: {task['task_name']}",
              result=result)

        if not self.results:
          self.results = TaskResultList.parse_obj([
            task_result
          ])
        else:
          self.results.__root__.append(task_result)

        # Step 3: Store the result in FAISS
        self.task_result_retriever.add_documents([
          Document(
          page_content=result,
          metadata={"task_name": task["task_name"], "task_id": task["task_id"]})
        ])

        # Step 4: Create new tasks and reprioritize task list
        new_tasks = self.get_next_task(
          result,
          task["task_name"], [t["task_name"] for t in self.task_list],
          objective
        )

        for new_task in new_tasks:
          self.task_id_counter += 1
          new_task.update({"task_id": self.task_id_counter})
          self.add_task(new_task)
        self.task_list = deque(
          self.prioritize_tasks(
            this_task_id, list(self.task_list), objective
          )
        )

      num_iters += 1
      if (self.max_iterations is not None and num_iters >= self.max_iterations):
        logger.info("****TASK ENDING. Iterations: %s *****", num_iters)
        break

    if self.task_list:
      self.task_vectorstore = _create_default_vectorstore(embeddings=self.task_embeddings)
      self.task_vectorstore.add_texts(
        texts = [task["task_name"] for task in self.task_list],
        metadatas=[
          {"task_id": task["task_id"], "task_name": task["task_name"]} for task in self.task_list],
        id = [task["task_id"] for task in self.task_list],
      )
    return { }


  @classmethod
  def from_llm_and_vectorstores(
    cls,
    llm: BaseLanguageModel,
    verbose: bool = False,
    execution_chain: Optional[AgentExecutor] = None,
    max_iterations: Optional[int] = 5,
    max_task_list_num: Optional[int] = 5,
    task_embeddings: Optional[Embeddings] = None,
    task_vectorstore: Optional[VectorStore] = None,
    task_result_vectorstore: Optional[VectorStore] = None,
    task_result_decay_rate: float =  0.01,
    task_result_k: int = 1,
    callbacks: Optional[Callbacks] = None,
    **kwargs: Dict[str, Any],
  ) -> "PersistentBabyAGI":
    """Initialize the PersistentBabyAGI Controller."""
    # Instantiate Task objects
    task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
    task_prioritization_chain = TaskPrioritizationChain.from_llm(
      llm, verbose=verbose
    )

    # Instantiate Task Vectorstores and retrievers
    if not task_embeddings:
      task_embeddings = _create_default_task_embeddings()
    if not task_vectorstore:
      task_vectorstore = _create_default_vectorstore(task_embeddings)
    if not task_result_vectorstore:
      task_result_vectorstore = _create_default_vectorstore(task_embeddings)

    task_result_retriever = TimeWeightedVectorStoreRetriever(
      vectorstore=task_result_vectorstore,
      decay_rate=task_result_decay_rate,
      k=task_result_k)

    return cls(
      task_creation_chain=task_creation_chain,
      task_prioritization_chain=task_prioritization_chain,
      execution_chain=execution_chain,
      task_embeddings=task_embeddings,
      task_vectorstore=task_vectorstore,
      task_result_vectorstore=task_result_vectorstore,
      task_result_retriever=task_result_retriever,
      max_task_list_num=max_task_list_num,
      max_iterations=max_iterations,
      callbacks=callbacks,
      **kwargs,
    )
