"""Boilerplate lambda function and then some..."""
import os
from typing import Any, Dict, List
import logging
import json
import faiss

from jsonschema import validate

from langchain.tools import BaseTool
from langchain.callbacks.stdout import StdOutCallbackHandler

from langchain.agents import load_tools
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.experimental.autonomous_agents.baby_agi import (
  BabyAGI
)


#TODO - Add your OpenAI API key here or in cdk.json
# os.environ["OPENAI_API_KEY"] = ""

# Boiler plate env variables
deployment_environment = os.getenv("ENVIRONMENT", "dev")
region = os.getenv("REGION_NAME", None)
log_level = os.getenv("LOG_LEVEL", "INFO")

# Search results number
search_results_num = int(os.getenv("SEARCH_RESULTS_NUM", "5"))
# Wikipedia search results
wikipedia_results_num = int(os.getenv("WIKIPEDIA_RESULTS_NUM", "1"))
# OpenAI default model name
openai_model_name = os.getenv("OPENAI_MODEL_NAME", "text-davinci-003")

llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

agent_max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
agent_max_task_list_num = int(os.getenv("AGENT_MAX_TASK_NUM", "5"))

chrome_driver_dir = os.getenv("CHROME_DRIVER_DIR")

_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "Input objective schema",
  "description": "The fields required for BabyAGI's objective",
  "required": ["arguments"],
  "properties": {
    "arguments": {
      "type": "object",
      "required": ["input"],
      "properties": {
        "input": {
          "type": "object",
          "required": ["search_text"],
          "properties": {
            "search_text": {
              "$id": "#/arguments/input/search_text",
              "type": "string",
              "title": "search_text",
            },
          }
        }
      },
    }
  }
}

logger = logging.getLogger()
logger.setLevel(log_level)

# Instantiate LLM embeddings
EMBEDDINGS = OpenAIEmbeddings()

# Instantiate callback
callback_handler = [StdOutCallbackHandler()]

# Instantiate LLM model
llm = OpenAI(
  model_name=openai_model_name,
  temperature=llm_temperature,
  callbacks=callback_handler)

# Example tools - Loading DuckDuckGo and Wikipedia. Add more tools as you please..
tools: List[BaseTool] = load_tools(
  ["ddg-search", "wikipedia"],
  max_results=search_results_num,
  top_k_results=wikipedia_results_num)

# Add a todo Tool
todo_prompt = PromptTemplate.from_template(
  """You are a planner who is an expert at coming up with a todo list for a given objective.
  Come up with a todo list for this objective: {objective}"""
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

tools.append(
  Tool(
    name="TODO",
    func=todo_chain.run,
    description="""useful for when you need to come up with todo lists.
      Input: an objective to create a todo list for.
      Output: a todo list for that objective. 
      Please be very clear what the objective is!""")
)

# Instantiate execution agent
zero_agent_prompt = ZeroShotAgent.create_prompt(
  tools,
  prefix="""You are an AI who performs one task based on the following objective: {objective}.
Take into account these previously completed tasks: {context}.""",
  suffix="""Question: {task}
{agent_scratchpad}""",
  input_variables=["objective", "task", "context", "agent_scratchpad"],
)

tool_names = [tool.name for tool in tools]
llm_chain = LLMChain(llm=llm, prompt=zero_agent_prompt)
zero_shot_agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
  agent=zero_shot_agent, tools=tools, verbose=True
)

def lambda_handler(event: Dict[str, Any], context):
  """"Lambda handler"""

  # Validate incoming event's schema
  validate(event, _schema)

  logger.debug(json.dumps(event, indent=2))
  search_text = event.get("arguments").get("input").get("search_text")

  if not search_text:
    raise ValueError("search text empty")

  # Instantiate vectorstore for tasks
  embedding_size: int = len(EMBEDDINGS.embed_query("test"))
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore: FAISS = FAISS(EMBEDDINGS.embed_query, index, InMemoryDocstore({}), {})

  # Instantiate and run the agent
  agent = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=True
  )
  # The agent returns an empty dict. Go to Cloudwatch to view the agent's logs
  resp = agent({"objective": search_text})

  return resp
