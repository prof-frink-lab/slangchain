"""Language Agent Tree Search"""
from __future__ import annotations
from collections import deque, defaultdict
import math
import logging
import asyncio
import json
from typing import Dict, List, Sequence, Optional, Any
from typing_extensions import TypedDict

from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, chain as as_runnable
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_anthropic.output_parsers import ToolsOutputParser

from langchain.schema import ChatResult
from langchain.chains.base import Chain
from langchain.tools.base import BaseTool

from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph

from slangchain.graphs.anthropic.schemas import Reflection
from slangchain.tools.reflection.tool import ReflectionTool
from slangchain.llms.chat_anthropic_tools import ChatAnthropicTools

logger = logging.getLogger(__name__)


class CandidatesGenerator(BaseModel):
  """InitalResponseChain"""
  completions_num: Optional[int] = Field(default = 5)
  completions_temperature: Optional[float] = Field(default = 0.0)

  llm: ChatAnthropicTools

  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True


  async def _generate_candidates_async(
    self,
    llm: ChatAnthropicTools,
    messages: ChatPromptValue,
    config: RunnableConfig) -> List[ChatResult]:
    """_generate_candidates_async"""
    generated_messages = []

    bound_kwargs = self.llm.model.kwargs
    tasks = []
    for _ in range(self.completions_num):
      task = llm.agenerate(
        [messages.to_messages()],
        callbacks = config["callbacks"],
        run_name = "CandidatesGenerator",
        temperature = self.completions_temperature,
        **bound_kwargs
      )
      tasks.append(task)

    results = await asyncio.gather(*tasks)
    for chat_result in results:
      if chat_result.generations:
        generated_messages.extend([gen.message for gen in chat_result.generations[0]])
    return generated_messages


  def generate_candidates(
    self,
    messages: ChatPromptValue,
    config: RunnableConfig) -> List[ChatResult]:
    """generate_candidates"""

    generated_messages = []
    async def run_async_task():
      return await self._generate_candidates_async(self.llm, messages, config)

    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
      # Run the asynchronous task in the event loop
      generated_messages = loop.run_until_complete(run_async_task())
    finally:
      # Close the event loop
      loop.close()

    return generated_messages


  @classmethod
  def from_tools(
    cls,
    tools: Sequence[BaseTool],
    llm: ChatAnthropicTools,
    completions_num: Optional[int] = 5,
    completions_temperature: Optional[int] = 0.0,
  ) -> "CandidatesGenerator":
    """Construct CandidatesGenerator from params."""

    llm.bind_tools(tools=tools)

    return cls(
      llm = llm,
      completions_num = completions_num,
      completions_temperature = completions_temperature,
    )


@as_runnable
def initial_answer_chain(
    inputs: Dict[str, Any],
    llm: ChatAnthropicTools,
    tools: Sequence[BaseTool]) -> Dict:
  """initial_answer_chain"""
  prompt_template = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "You are an AI assistant.",
      ),
      ("user", "{input}"),
      MessagesPlaceholder(variable_name="messages", optional=True),
    ]
  )

  llm.bind_tools(tools)
  chain = prompt_template | llm
  result = chain.invoke(inputs)

  return result


@as_runnable
def reflection_chain(inputs, llm: ChatAnthropicTools, retries: int) -> Reflection:
  """Reflection chain"""
  attempt = 0
  result = None
  reflection = None
  error_msg = ""
  reflection_tool = ReflectionTool()

  prompt = ChatPromptTemplate.from_messages(
    [
    (
      "system",
      (
        (
          f"Use the {reflection_tool.name} tool to reflect and grade the"
          " assistant response to the user question below."
        )
      ),
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="candidate"),
    ]
  )

  llm.bind_tools(tools = [ReflectionTool()])

  llm_chain = prompt | llm

  tool_parser = ToolsOutputParser(pydantic_schemas=[Reflection])

  while attempt < retries:
    try:
      result = llm_chain.invoke(inputs)
      tool_choices = tool_parser.invoke(result)
      reflection = tool_choices[0]
      if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
      return reflection
    except Exception as exp:
      error_msg = f"Error: {exp}, Result: {result}"
      logger.debug("Reflection error: %s\n", result)
      attempt += 1

  if not reflection:
    reflection = Reflection(
      reflections = (
        f'Unable to reflect and grade assistant response due to the following: {error_msg}'
      ),
      score = 0,
      found_solution = False
    )

  return reflection


@as_runnable
def generate_candidates_chain(
  inputs,
  llm: ChatAnthropicTools,
  tools: Sequence[BaseTool],
  completions_num: int,
  completions_temperature: int
  ) -> Dict:
  """Generate candidates chain"""
  prompt_template = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "You are an AI assistant.",
      ),
      ("user", "{input}"),
      MessagesPlaceholder(variable_name="messages", optional=True),
    ]
  )

  generate_candidates = CandidatesGenerator.from_tools(
    tools = tools,
    llm = llm,
    completions_num = completions_num,
    completions_temperature = completions_temperature)

  chain = prompt_template | generate_candidates.generate_candidates

  result = chain.ainvoke(input = inputs)

  return result


class Node:
  """Node"""
  def __init__(
    self,
    messages: List[BaseMessage],
    reflection: Reflection,
    parent: Optional[Node] = None,
  ):
    self.messages = messages
    self.parent = parent
    self.children = []
    self.value = 0
    self.visits = 0
    self.reflection = reflection
    self.depth = parent.depth + 1 if parent is not None else 1
    self._is_solved = reflection.found_solution if reflection else False
    if self._is_solved:
      self._mark_tree_as_solved()
    self.backpropagate(reflection.normalized_score)

  def __repr__(self) -> str:
    return (
      f"<Node value={self.value}, visits={self.visits},"
      f" solution={self.messages} reflection={self.reflection}/>"
    )

  @property
  def is_solved(self):
    """If any solutions exist, we can end the search."""
    return self._is_solved


  @is_solved.setter # when you do Stock.name = x, it will call this function
  def is_solved(self, is_solved):
    self._is_solved = is_solved


  @property
  def is_terminal(self):
    """is_terminal"""
    return not self.children


  @property
  def best_child(self):
    """Select the child with the highest UCT to search next."""
    if not self.children:
      return None
    all_nodes = self._get_all_children()
    return max(all_nodes, key=lambda child: child.upper_confidence_bound())


  @property
  def best_child_score(self):
    """Return the child with the highest value."""
    if not self.children:
      return None
    return max(self.children, key=lambda child: int(child.is_solved) * child.value)


  @property
  def height(self) -> int:
    """Check for how far we've rolled out the tree."""
    if self.children:
      return 1 + max([child.height for child in self.children])
    return 1


  def upper_confidence_bound(self, exploration_weight=1.0):
    """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
    if self.parent is None:
      raise ValueError("Cannot obtain UCT from root node")
    if self.visits == 0:
      return self.value
    # Encourages exploitation of high-value trajectories
    average_reward = self.value / self.visits
    # Encourages exploration of less-visited trajectories
    exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
    return average_reward + exploration_weight * exploration_term


  def backpropagate(self, reward: float):
    """Update the score of this node and its parents."""
    node = self
    while node:
      node.visits += 1
      node.value = (node.value * (node.visits - 1) + reward) / node.visits
      node = node.parent


  def get_messages(self, include_reflections: bool = True):
    """get_messages"""
    if include_reflections:
      return self.messages + [self.reflection.as_message()]
    return self.messages


  def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
    """Get messages representing this search branch."""
    messages = []
    node = self
    while node:
      messages.extend(
        node.get_messages(include_reflections=include_reflections)[::-1]
      )
      node = node.parent
    # Reverse the final back-tracked trajectory to return in the correct order
    return messages[::-1]  # root solution, reflection, child 1, ...


  def _get_all_children(self):
    all_nodes = []
    nodes = deque()
    nodes.append(self)
    while nodes:
      node = nodes.popleft()
      all_nodes.extend(node.children)
      for n in node.children:
        nodes.append(n)
    return all_nodes


  def get_best_solution(self):
    """Return the best solution from within the current sub-tree."""
    all_nodes = [self] + self._get_all_children()
    best_node = max(
      all_nodes,
      # We filter out all non-terminal, non-solution trajectories
      key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
    )
    return best_node


  def _mark_tree_as_solved(self):
    parent = self.parent
    while parent:
      parent.is_solved = True
      parent = parent.parent


class TreeState(TypedDict):
  """Tree State"""
  # The full tree
  root: Node
    # The original input
  input: str


class LATS(Chain):
  """Lateral Agent Tree Search"""
  model_name: str
  tree_height: Optional[int] = Field(default = 5)

  tools: Sequence[BaseTool]
  tool_executor: ToolExecutor
  llm: ChatAnthropicTools
  reflection_retries: int
  tool_parser: ToolsOutputParser

  candidate_completions_num: int
  candidate_completions_temperature: int

  workflow: Optional[StateGraph] = Field(default = None)

  input_key: str = "input"  #: :meta private:
  output_key: str = "output"  #: :meta private:


  @property
  def input_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return [self.input_key]

  @property
  def output_keys(self) -> List[str]:
    """Will be whatever keys the prompt expects.

    :meta private:
    """
    return [self.output_key]


  class Config:
    """BaseModel config"""
    extra = Extra.forbid
    arbitrary_types_allowed = True


  @root_validator()
  def validate_class_objects(cls, values: Dict) -> Dict:
    """Validate that chains are all single input/output."""
    if "claude-3" not in values["model_name"]:
      raise TypeError("model_name must start with claude-3")
    if not values["tools"]:
      return ValueError("tools empty")
    if not all(isinstance(node_tool, BaseTool) for node_tool in values["tools"]):
      raise TypeError("tools all must be of BaseTool type")
    return values


  def generate_initial_response(self, state: TreeState) -> Dict:
    """generate initial response"""
    result = initial_answer_chain.invoke(
      {
        "input": state["input"]
      },
      llm = self.llm,
      tools = self.tools)
    parsed = self.tool_parser.invoke(result)

    logger.debug("\ngenerate_initial_response parsed: %s\n", parsed)

    tool_responses = self.tool_executor.batch(
      [ToolInvocation(tool=r["name"], tool_input=r["args"]) for r in parsed]
    )

    logger.debug("\ngenerate_initial_response tool_responses: %s\n", tool_responses)

    output_messages = [result] + [
        ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        for resp, tool_call in zip(tool_responses, parsed)
    ]

    reflection = reflection_chain.invoke(
      {"input": state["input"], "candidate": output_messages},
      llm = self.llm,
      retries = self.reflection_retries
    )

    logger.debug("\ngenerate_initial_response reflection: %s", reflection)

    root = Node(output_messages, reflection=reflection)

    return {
      **state,
      "root": root,
    }


  def expand(self, state: TreeState, config: RunnableConfig) -> TreeState:
    """expand"""
    root = state["root"]
    best_candidate: Node = root.best_child if root.children else root
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    logger.debug('\n%s\n', {"input": state["input"], "messages": messages})
    new_candidates = asyncio.run(generate_candidates_chain.invoke(
       {"input": state["input"], "messages": messages},
       config = config,
       llm = self.llm,
       tools = self.tools,
       completions_num = self.candidate_completions_num,
       completions_temperature = self.candidate_completions_temperature
    ))

    logger.debug("new_candidates: %s", new_candidates)

    parsed = self.tool_parser.batch(new_candidates)

    logger.debug("parsed: %s", parsed)

    flattened = [
      (i, tool_call)
      for i, tool_calls in enumerate(parsed)
      for tool_call in tool_calls
    ]
    tool_responses = self.tool_executor.batch(
      [
        ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
        for _, tool_call in flattened
      ]
    )
    collected_responses = defaultdict(list)
    for (i, tool_call), resp in zip(flattened, tool_responses):
      collected_responses[i].append(
        ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
      )
    output_messages = []
    for i, candidate in enumerate(new_candidates):
      output_messages.append([candidate] + collected_responses[i])

    logger.debug("output_messages: %s", output_messages)

    # Reflect on each candidate
    # For tasks with external validation, you'd add that here.
    reflections = reflection_chain.batch(
        [{"input": state["input"], "candidate": msges} for msges in output_messages],
        llm = self.llm,
        retries = self.reflection_retries,
        config = config
    )

    logger.debug("reflections: %s", reflections)

    # Grow tree
    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    # We have already extended the tree directly, so we just return the state
    return state


  def _should_loop(self, state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
      return END
    if root.height > self.tree_height:
      return END
    return "expand"


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""

    self.workflow.add_node("start", self.generate_initial_response)
    self.workflow.add_node("expand", self.expand)
    self.workflow.set_entry_point("start")


    self.workflow.add_conditional_edges(
      "start",
      # Either expand/rollout or finish
      self._should_loop,
    )
    self.workflow.add_conditional_edges(
      "expand",
      # Either continue to rollout or finish
      self._should_loop,
    )

    return self.workflow


  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:

    message = inputs[self.input_key]
    self.workflow = StateGraph(TreeState)
    self.init_workflow_nodes()
    graph = self.workflow.compile()

    step = None

    for step in graph.stream({"input": message}):
      step_name, step_state = next(iter(step.items()))
      logger.info("\n%s", step_name)
      logger.info("\nrolled out: %s", step_state["root"].height)
      logger.info("---\n")

    return {self.output_key: step}


  @classmethod
  def from_tools(
    cls,
    tools: Sequence[BaseTool],
    model_name: Optional[str] = "claude-3-haiku-20240307",
    reflection_retries: Optional[int] = 3,
    candidate_completions_num: Optional[int] = 5,
    candidate_completions_temperature: Optional[int] = 0.0,
    tree_height: Optional[int] = 5,
  ) -> "LATS":
    """Construct an LATS from tools."""

    tool_executor = ToolExecutor(tools = tools)

    llm = ChatAnthropicTools(model_name = model_name)
    llm = ChatAnthropicTools(model_name = model_name)
    llm = ChatAnthropicTools(model_name = model_name)

    tool_parser = ToolsOutputParser()

    return cls(
      model_name = model_name,
      tools = tools,
      reflection_retries = reflection_retries,
      tool_executor = tool_executor,
      llm = llm,
      candidate_completions_num = candidate_completions_num,
      candidate_completions_temperature = candidate_completions_temperature,
      tool_parser = tool_parser,
      tree_height = tree_height
    )
