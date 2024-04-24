"""Web Navigation"""
from typing import (
  Dict, List, Union, Optional, Any
)
import logging
import os
import re
import asyncio
import platform
import base64
from playwright.async_api import Page

from playwright.async_api import async_playwright



from langchain_core.messages import (
  HumanMessage,
  AIMessage,
  ChatMessage,
  SystemMessage
)
from langchain_core.prompts import (
  PromptTemplate,
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.runnables import chain as chain_decorator
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_anthropic import ChatAnthropic

from langchain.chains.base import Chain

from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

from slangchain.graphs.anthropic.schemas import WebNavigationAgentState as AgentState

SYSTEM_MESSAGE: str = (
  "Imagine you are a robot browsing the web, just like humans. Now you need to complete a task."
  " In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts."
  " This screenshot will\nfeature Numerical Labels placed in the TOP LEFT corner of each Web Element."
  " Carefully analyze the visual\ninformation to identify the Numerical Label corresponding to the Web Element that requires interaction,"
  " then follow\nthe guidelines and choose one of the following actions:\n\n1. Click a Web Element."
  "\n2. Delete existing content in a textbox and then type content.\n3. Scroll up or down.\n4. Wait"
  " \n5. Go back\n7. Return to google to start over.\n8. Respond with the final answer"
  "\n\nCorrespondingly, Action should STRICTLY follow the format:\n\n- Click [Numerical_Label]"
  " \n- Type [Numerical_Label]; [Content] \n- Scroll [Numerical_Label or WINDOW]; [up or down]"
  " \n- Wait \n- GoBack\n- Google\n- ANSWER; [content]\n\nKey Guidelines You MUST follow:"
  "\n\n* Action guidelines *\n1) Execute only one action per iteration."
  "\n2) When clicking or typing, ensure to select the correct bounding box."
  "\n3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n"
  "* Web Browsing Guidelines *\n1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages"
  "\n2) Select strategically to minimize time wasted."
)

ACTIONS_MESSAGE: str = (
  "\n\nYou MUST only strictly follow the format:"
  "\n\nThought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}"
  "\nAction: {{One Action format you choose}}"
  "\nThen the User will provide:"
  "\nObservation: {{A labeled screenshot Given by User}}\n"
)

IMAGE_TEMPLATE: Dict = {
  'url': 'data:image/png;base64,{img}'
}

BBOX_DESCRIPION_TEMPLATE: str = '{bbox_descriptions}'

INPUT_TEMPLATE: str = '{input}'

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate(
  input_variables = ['bbox_descriptions', 'img', 'input'],
  input_types = {'scratchpad': List[Union[AIMessage, HumanMessage, ChatMessage, SystemMessage]]},
  partial_variables = {'scratchpad': []},
  messages=[
    SystemMessagePromptTemplate(
      prompt = PromptTemplate(
        input_variables = [],
        template = SYSTEM_MESSAGE)),
    HumanMessagePromptTemplate(
      prompt = PromptTemplate(
        input_variables = [],
        template = ACTIONS_MESSAGE)),
    MessagesPlaceholder(
      variable_name = 'scratchpad', optional=True),
      HumanMessagePromptTemplate(
        prompt = [
          ImagePromptTemplate(
            input_variables = ['img'],
            template = {'url': 'data:image/png;base64,{img}'}),
          PromptTemplate(
            input_variables = ['bbox_descriptions'],
            template = '{bbox_descriptions}'),
          PromptTemplate(
            input_variables = ['input'],
            template = '{input}')])
    ]
)


@chain_decorator
async def mark_page(page: Page):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  js_file_path = os.path.join(current_dir, "mark_page.js")

  with open(js_file_path, encoding = "utf-8") as f:
    mark_page_script = f.read()
  await page.evaluate(mark_page_script)
  for _ in range(10):
    try:
      bboxes = await page.evaluate("markPage()")
      break
    except Exception:
      # May be loading...
      asyncio.sleep(3)
  screenshot = await page.screenshot()
  # Ensure the bboxes don't follow us around
  await page.evaluate("unmarkPage()")
  return {
    "img": base64.b64encode(screenshot).decode(),
    "bboxes": bboxes,
  }


class WebVoyager(Chain):
  """Web Voyager"""

  llm: ChatAnthropic
  prompt: ChatPromptTemplate = Field(default = PROMPT)
  recursion_limit: Optional[int] = Field(default = 100)
  headless_flag: Optional[bool] = Field(default = False)
  agent: Optional[Runnable] = Field(default = None)
  workflow: Optional[StateGraph] = Field(default = None)
  input_key: str = "input"  #: :meta private:
  output_key: str = "output"  #: :meta private:


  class Config:
    """BaseModel config"""
    extra = Extra.allow
    arbitrary_types_allowed = True


  @root_validator()
  def validate_class_objects(cls, values: Dict) -> Dict:
    """Validate that chains are all single input/output."""
    if not isinstance(values["llm"], ChatAnthropic):
      raise TypeError("llm must be of instance ChatAnthropic")
    return values


  def _create_agent(self) -> Runnable:
    """create_agent"""
    self.agent = self._annotate | RunnablePassthrough.assign(
      prediction = self._format_descriptions | PROMPT | self.llm | StrOutputParser() | self._parse
    )
    return self.agent


  def init_workflow_nodes(self) -> StateGraph:
    """init workflow nodes"""
    self.workflow = StateGraph(AgentState)
    self.agent = self._create_agent()
    self.workflow.add_node("agent", self.agent)
    self.workflow.set_entry_point("agent")

    self.workflow.add_node("update_scratchpad", self._update_scratchpad)
    self.workflow.add_edge("update_scratchpad", "agent")

    tools = {
        "Click": self._click,
        "Type": self._type_text,
        "Scroll": self._scroll,
        "Wait": self._wait,
        "GoBack": self._go_back,
        "Google": self._to_google,
    }

    for node_name, tool in tools.items():
      self.workflow.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
      )
      # Always return to the agent (by means of the update-scratchpad node)
      self.workflow.add_edge(node_name, "update_scratchpad")

    self.workflow.add_conditional_edges("agent", self._select_tool)

    return self.workflow


  async def _acall(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:

    question = inputs[self.input_key]
    self.init_workflow_nodes()
    graph = self.compile_graph()

    result : Dict[str, Any] = {}

    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = await browser.chromium.launch(
      headless= self.headless_flag,
      args=None)
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")

    steps = []
    event_stream = graph.astream(
      {
        "page": page,
        "input": f"\n {question}",
        "scratchpad": [AIMessage(content="1. Start")],
      },
      {
        "recursion_limit": self.recursion_limit,
      },)

    async for event in event_stream:
      # We'll display an event stream here
      if "agent" not in event:
        continue
      pred = event["agent"].get("prediction") or {}
      action = pred.get("action")
      action_input = pred.get("args")
      steps.append(f"{len(steps) + 1}. {action}: {action_input}")
      logger.info("Steps: %s", "\n".join(steps))
      if "ANSWER" in action:
        result = action_input[0]
        break

    return {self.output_key: result}


  def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
  ) -> Dict[str, Any]:
    raise NotImplementedError("call not implemented")


  @classmethod
  def from_llm(
    cls,
    llm: ChatAnthropic,
    recursion_limit: Optional[int] = 100,
    headless_flag: Optional[bool] = False
  ) -> "WebVoyager":
    """Construct an WebVoyager from an LLM."""

    prompt = PROMPT

    if not headless_flag:
      headless_flag = False

    return cls(
      llm = llm,
      prompt = prompt,
      recursion_limit = recursion_limit,
      headless_flag = headless_flag
    )


  def compile_graph(self) -> Pregel:
    """compile graph"""
    graph = self.workflow.compile()
    return graph


  def _select_tool(self, state: AgentState):
    """_select_tool"""
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == "ANSWER":
      return END
    if action == "retry":
      return "agent"
    return action


  async def _click(self, state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
      return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
      bbox = state["bboxes"][bbox_id]
    except Exception:
      return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    res = await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.
    return f"Clicked {bbox_id}"


  async def _type_text(self, state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
      return (
        f"Failed to type in element from bounding box labeled as number {type_args}"
      )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


  async def _scroll(self, state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
      return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
      # Not sure the best value for this:
      scroll_amount = 500
      scroll_direction = (
        -scroll_amount if direction.lower() == "up" else scroll_amount
      )
      await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
      # Scrolling within a specific element
      scroll_amount = 200
      target_id = int(target)
      bbox = state["bboxes"][target_id]
      x, y = bbox["x"], bbox["y"]
      scroll_direction = (
        -scroll_amount if direction.lower() == "up" else scroll_amount
      )
      await page.mouse.move(x, y)
      await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


  async def _wait(self, state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


  async def _go_back(self, state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


  async def _to_google(self, state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."


  async def _annotate(self, state: AgentState):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


  def _format_descriptions(self, state: AgentState):
    """format descriptions"""
    labels = []

    for i, bbox in enumerate(state["bboxes"]):

      text = bbox.get("ariaLabel") or ""
      if not text.strip():
        text = bbox["text"]
      el_type = bbox.get("type")
      labels.append(f'{i} (<{el_type}/>): "{text}"')

    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)

    return {**state, "bbox_descriptions": bbox_descriptions}


  def _parse(self, text: str) -> dict:
    """parse text to action dict"""

    action_prefix = "Action: "

    if not text.strip().split("\n")[-1].startswith(action_prefix):
      return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}

    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)

    if len(split_output) == 1:
      action, action_input = split_output[0], None
    else:
      action, action_input = split_output
    action = action.strip()

    if action_input is not None:
      action_input = [
        inp.strip().strip("[]") for inp in action_input.strip().split(";")
      ]

    return {"action": action, "args": action_input}


  def _update_scratchpad(self, state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""

    old = state.get("scratchpad")
    if old:
      txt = old[0].content
      last_line = txt.rsplit("\n", 1)[-1]
      step = int(re.match(r"\d+", last_line).group()) + 1
    else:
      txt = "Previous action observations:\n"
      step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [AIMessage(content=txt)]}


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
