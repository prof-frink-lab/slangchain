"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from __future__ import annotations
from abc import abstractmethod
import logging
import re
from typing import (
  Any,
  Callable,
  Dict,
  List,
  NamedTuple,
  Optional,
  Sequence,
  Tuple,
  Union
)

from langchain_core.pydantic_v1 import Field, root_validator

from slangchain.agents.constrained_mrkl.prompt import (
    FORMAT_INSTRUCTIONS, PREFIX, SUFFIX, INPUT_VARIABLES
)

from transformers import GPT2TokenizerFast

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage
)
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from langchain.agents.agent import AgentOutputParser
from langchain.agents import BaseSingleActionAgent
from langchain.agents.mrkl.output_parser import MRKLOutputParser

FINAL_ANSWER_ACTION = "Final Answer:"

logger = logging.getLogger(__name__)

class ChainConfig(NamedTuple):
  """Configuration for chain to use in MRKL system.

  Args:
    action_name: Name of the action.
    action: Action function to call.
    action_description: Description of the action.
  """

  action_name: str
  action: Callable
  action_description: str


def get_action_and_input(llm_output: str) -> Tuple[str, str]:
  """Parse out the action and input from the LLM output.

  Note: if you're specifying a custom prompt for the ZeroShotAgent,
  you will need to ensure that it meets the following Regex requirements.
  The string starting with "Action:" and the following string starting
  with "Action Input:" should be separated by a newline.
  """
  if FINAL_ANSWER_ACTION in llm_output:
    return "Final Answer", llm_output.split(FINAL_ANSWER_ACTION)[-1].strip()
    # \s matches against tab/newline/whitespace
  regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
  match = re.search(regex, llm_output, re.DOTALL)
  if not match:
    return "Final Answer", llm_output
  action = match.group(1).strip()
  action_input = match.group(2)
  return action, action_input.strip(" ").strip('"')



class ThriftyAgent(BaseSingleActionAgent):
  """Class responsible for calling the language model and deciding the action.

  This is driven by an LLMChain. The prompt in the LLMChain MUST include
  a variable called "agent_scratchpad" where the agent can put its
  intermediary work.
  """
  tokenizer: Optional[GPT2TokenizerFast] = GPT2TokenizerFast.from_pretrained("gpt2")
  llm_chain: LLMChain
  output_parser: AgentOutputParser
  allowed_tools: Optional[List[str]] = None
  agent_scratchpad_token_limit: Optional[int] = None

  def get_allowed_tools(self) -> Optional[List[str]]:
    return self.allowed_tools

  class Config:
    """BaseTool config class"""
    arbitrary_types_allowed = True

  @property
  def return_values(self) -> List[str]:
    return ["output"]

  def _fix_text(self, text: str) -> str:
    """Fix the text."""
    raise ValueError("fix_text not implemented for this agent.")

  @property
  def _stop(self) -> List[str]:
    return [
      f"\n{self.observation_prefix.rstrip()}",
      f"\n\t{self.observation_prefix.rstrip()}",
    ]

  def _construct_scratchpad(
    self, intermediate_steps: List[Tuple[AgentAction, str]]
  ) -> Union[str, List[BaseMessage]]:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    if self.agent_scratchpad_token_limit:
      thoughts_list = []
      thoughts = ""
      for action, observation in reversed(intermediate_steps):
        new_thought = f"{action.log}\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        num_thought_token_len = self._get_token_count(new_thought)
        thoughts_token_len = sum(self._get_token_count(i) for i in thoughts_list)
        if thoughts_token_len + num_thought_token_len > self.agent_scratchpad_token_limit:
          logger.warning("Trimming scratchpad as token length %s will exceed a token limit of %s", \
                         thoughts_token_len + num_thought_token_len, \
                          self.agent_scratchpad_token_limit)
          break
        thoughts_list.append(new_thought)
      thoughts = "".join(reversed(thoughts_list))
    else:
      for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
    return thoughts

  def plan(
    self,
    intermediate_steps: List[Tuple[AgentAction, str]],
    callbacks: Callbacks = None,
    **kwargs: Any
  ) -> Union[AgentAction, AgentFinish]:
    """Given input, decided what to do.

    Args:
      intermediate_steps: Steps the LLM has taken to date,
        along with observations
      **kwargs: User inputs.

    Returns:
      Action specifying what tool to use.
    """
    full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
    full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
    return self.output_parser.parse(full_output)

  async def aplan(
    self,
    intermediate_steps: List[Tuple[AgentAction, str]],
    callbacks: Callbacks = None,
    **kwargs: Any
  ) -> Union[AgentAction, AgentFinish]:
    """Given input, decided what to do.

    Args:
      intermediate_steps: Steps the LLM has taken to date,
        along with observations
      **kwargs: User inputs.

    Returns:
      Action specifying what tool to use.
    """
    full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
    full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
    return self.output_parser.parse(full_output)

  def get_full_inputs(
    self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
  ) -> Dict[str, Any]:
    """Create the full inputs for the LLMChain from intermediate steps."""
    thoughts = self._construct_scratchpad(intermediate_steps)
    new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
    full_inputs = {**kwargs, **new_inputs}
    return full_inputs

  def _get_token_count(self, input_str) -> int:
    return len(self.tokenizer.encode(input_str))

  @property
  def input_keys(self) -> List[str]:
    """Return the input keys.

    :meta private:
    """
    return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

  @root_validator()
  def validate_prompt(cls, values: Dict) -> Dict:
    """Validate that prompt matches format."""
    prompt = values["llm_chain"].prompt
    if "agent_scratchpad" not in prompt.input_variables:
      logger.warning(
        "`agent_scratchpad` should be a variable in prompt.input_variables."
        " Did not find it, so adding it at the end."
      )
      prompt.input_variables.append("agent_scratchpad")
      if isinstance(prompt, PromptTemplate):
        prompt.template += "\n{agent_scratchpad}"
      elif isinstance(prompt, FewShotPromptTemplate):
        prompt.suffix += "\n{agent_scratchpad}"
      else:
        raise ValueError(f"Got unexpected prompt type {type(prompt)}")
    return values

  @property
  @abstractmethod
  def observation_prefix(self) -> str:
    """Prefix to append the observation with."""

  @property
  @abstractmethod
  def llm_prefix(self) -> str:
    """Prefix to append the LLM call with."""

  @classmethod
  @abstractmethod
  def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
    """Create a prompt for this class."""

  @classmethod
  def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
    """Validate that appropriate tools are passed in."""
    pass

  @classmethod
  @abstractmethod
  def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
    """Get default output parser for this class."""

  @classmethod
  def from_llm_and_tools(
    cls,
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[AgentOutputParser] = None,
    **kwargs: Any,
  ) -> ThriftyAgent:
    """Construct an agent from an LLM and tools."""
    cls._validate_tools(tools)

    llm_chain = LLMChain(
      llm=llm,
      prompt=cls.create_prompt(tools),
      callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    _output_parser = output_parser or cls._get_default_output_parser()
    return cls(
      llm_chain=llm_chain,
      allowed_tools=tool_names,
      output_parser=_output_parser,
      **kwargs,
    )

  def return_stopped_response(
    self,
    early_stopping_method: str,
    intermediate_steps: List[Tuple[AgentAction, str]],
    **kwargs: Any,
  ) -> AgentFinish:
    """Return response when agent has been stopped due to max iterations."""
    if early_stopping_method == "force":
      # `force` just returns a constant string
      return AgentFinish(
        {"output": "Agent stopped due to iteration limit or time limit."}, ""
      )
    elif early_stopping_method == "generate":
      # Generate does one final forward pass
      thoughts = ""
      for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += (
          f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        )
      # Adding to the previous steps, we now tell the LLM to make a final pred
      thoughts += (
        "\n\nI now need to return a final answer based on the previous steps:"
      )
      new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
      full_inputs = {**kwargs, **new_inputs}
      full_output = self.llm_chain.predict(**full_inputs)
      # We try to extract a final answer
      parsed_output = self.output_parser.parse(full_output)
      if isinstance(parsed_output, AgentFinish):
        # If we can extract, we send the correct stuff
        return parsed_output
      else:
        # If we can extract, but the tool is not the final tool,
        # we just return the full output
        return AgentFinish({"output": full_output}, full_output)
    else:
      raise ValueError(
        "early_stopping_method should be one of `force` or `generate`, "
        f"got {early_stopping_method}"
      )

  def tool_run_logging_kwargs(self) -> Dict:
    return {
      "llm_prefix": self.llm_prefix,
      "observation_prefix": self.observation_prefix,
    }

class ConstrainedZeroShotAgent(ThriftyAgent):
  """Agent for the MRKL chain."""

  output_parser: AgentOutputParser = Field(default_factory=MRKLOutputParser)

  @classmethod
  def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
    return MRKLOutputParser()

  @property
  def _agent_type(self) -> str:
    """Return Identifier of agent type."""
    return "zero-shot-react-description"

  @property
  def observation_prefix(self) -> str:
    """Prefix to append the observation with."""
    return "Observation: "

  @property
  def llm_prefix(self) -> str:
    """Prefix to append the llm call with."""
    return "Thought:"

  @classmethod
  def create_prompt(
    cls,
    tools: Sequence[BaseTool],
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
  ) -> PromptTemplate:
    """Create prompt in the style of the zero shot agent.

      Args:
      tools: List of tools the agent will have access to, used to format the
              prompt.
      prefix: String to put before the list of tools.
      suffix: String to put after the list of tools.
      input_variables: List of input variables the final prompt will expect.

      Returns:
       A PromptTemplate with the template assembled from the pieces here.
    """
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
    if input_variables is None:
      input_variables = list(INPUT_VARIABLES)
    return PromptTemplate(template=template, input_variables=input_variables)

  @classmethod
  def from_llm_and_tools(
    cls,
    llm: BaseLLM,
    tools: Sequence[BaseTool],
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[AgentOutputParser] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    agent_scratchpad_token_limit: Optional[int] = 1000,
    **kwargs: Any,
  ) -> ThriftyAgent:
    """Construct an agent from an LLM and tools."""
    cls._validate_tools(tools)
    prompt = cls.create_prompt(
      tools,
      prefix=prefix,
      suffix=suffix,
      format_instructions=format_instructions,
      input_variables=input_variables)

    llm_chain = LLMChain(
      llm=llm,
      prompt=prompt,
      callback_manager=callback_manager)

    _output_parser = output_parser or cls._get_default_output_parser()
    tool_names = [tool.name for tool in tools]

    return cls(
      llm_chain=llm_chain,
      allowed_tools=tool_names,
      output_parser=_output_parser,
      agent_scratchpad_token_limit=agent_scratchpad_token_limit,
      **kwargs)

  @classmethod
  def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
    for tool in tools:
      if tool.description is None:
        raise ValueError(
          f"Got a tool {tool.name} without a description. For this agent, "
          f"a description must always be provided."
        )

  def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
    return get_action_and_input(text)
