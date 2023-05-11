# Examples
<br>

## Autonomous Agents

<br>

### Persistent BabyAGI

[PersistentBabyAGI](../../slangchain/autonomous_agents/baby_agi.py) that is build on top of [BabyAgi](https://github.com/hwchase17/langchain/blob/master/docs/use_cases/autonomous_agents/baby_agi_with_agent.ipynb).

In a nutshell, PersistentBabyAGI extends the functionality for BabyAGI, allowing you to persist the task and task results in the form of Vectorstores. You can pick up where you left off by saving/loading the task and task results Vectorstores.

The notebook can be found [here](./autonomous_agents/examples/baby_agi.ipynb).

<br>

### Persistent BabyAGI - Tools, Context Compression Searches and Constrained Zero Agents

This notebook showcases integrating tools and agents with [PersistentBabyAGI](../../slangchain/autonomous_agents/baby_agi.py) that is build on top of [BabyAgi](https://github.com/hwchase17/langchain/blob/master/docs/use_cases/autonomous_agents/baby_agi_with_agent.ipynb). PersistentBabyAGI extends the functionality for BabyAGI, allowing you to persist the task and task results in the form of Vectorstores. You can pick up where you left off by saving/loading the task and task results Vectorstores.

I added functionality to [ZeroShotAgent](https://github.com/hwchase17/langchain/blob/master/langchain/agents/mrkl/base.py) to address unexpected agent terminations when the agent's instruction narratives exceeded OpenAI's API token limit (~4000). This typically occurs during longer running actions that creates longer instructions when a larger number of tools is made available to the agent. The Open AI error occurs when the instructions attain the said limit.

[ConstrainedZeroShotAgent ](../../slangchain/agents/constrained_mrkl/base.py) works around OpenAI's API token limit by removing the oldest observations from the agent scratch pad so that the instructions parsed to OpenAI's API is below the set agent_scratchpad_token_limit parameter threshold.

The notebook can be found [here](./autonomous_agents/examples/baby_agi_and_constrained_agents.ipynb).

<br>


## Chains

<br>

### URL Contextual Compressions Retriever

[UrlCompressedDocSearchChain](../../slangchain/chains/url_compressed_doc_search/base.py) loads url web content of various types (i.e. html, pdf or json), and retrives the chunks of text that are most relevant to a search query.

The search uses [LangChain](https://github.com/hwchase17/langchain) [DocumentCompressorPipeline](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/contextual-compression.ipynb). The core idea is simple: given a specific query, we should be able to return only the documents relevant to that query, and only the parts of those documents that are relevant.

The notebook can be found [here](./chains/examples/url_compressed_doc_search.ipynb).

<br>

### Wikipedia Contextual Compressions Retriever

[WikipediaDocSearchChain](../../slangchain/chains/wikipedia_doc_search/base.py) loads Wikipedia pages based on a search query, and retrives the chunks of text that are most relevant to a search query.

The search uses [LangChain](https://github.com/hwchase17/langchain) [WikipediaLoader](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/document_loaders/examples/wikipedia.ipynb) to retrieve pages most relevant to the search query and [DocumentCompressorPipeline](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/contextual-compression.ipynb) to return parts of those documents that are relevant.

The notebook can be found [here](./chains/examples/wikipedia_doc_search.ipynb).

<br>

## Callbacks

<br>

### DynamoDB Callback

The notebook showcases how to use the [StreamingDynamoDBCallbackHandler](../../slangchain/callbacks/streaming_aws_ddb.py) class. The functionality streams callbacks to a AWS DyanmoDB table.

[LangChain](https://github.com/hwchase17/langchain) provides a callback system that allows you to hook into the various stages of your LLM application. This is useful for logging, [monitoring](https://python.langchain.com/en/latest/tracing.html), [streaming](https://python.langchain.com/en/latest/modules/models/llms/examples/streaming_llm.html), and other tasks.

The notebook can be found [here](./callbacks/examples/streaming_aws_ddb.ipynb).

<br>

## Document Loaders

<br>

### Web Content Document Loader

[UnstructuredURLFileLoader](../../document_loaders/url_file.py) loads url web content of various types (i.e. html, pdf or json)

The notebook can be found [here](./document_loaders/examples/url_file.ipynb)
