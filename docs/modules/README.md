# Examples
<br>

## Chains

<br>

### URL Contextual Compressions Retriever

[UrlCompressedDocSearchChain](../../slangchain/chains/url_compressed_doc_search/base.py) loads url web content of various types (i.e. html, pdf or json), and retrives the chunks of text that are most relevant to a search query.

The search uses [LangChain](https://github.com/hwchase17/langchain) [DocumentCompressorPipeline](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/contextual-compression.ipynb). The core idea is simple: given a specific query, we should be able to return only the documents relevant to that query, and only the parts of those documents that are relevant.

The notebook can be found [here](./chains/examples/url_compressed_doc_search.ipynb).

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
