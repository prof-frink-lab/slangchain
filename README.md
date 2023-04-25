#  SlangChain Repo  
This repo contains extended functionality based on [LangChain](https://github.com/hwchase17/langchain)
<br><br>
# Prerequisites

- [Python](https://www.python.org/downloads/)

- [Pip](https://pip.pypa.io/en/stable/installation/)

- ChromeDriver - An open source tool for automated testing of webapps across many browsers. It provides capabilities for navigating to web pages, user input, JavaScript execution, and more.  ChromeDriver is a standalone server that implements the W3C WebDriver standard.
  - [MacOS installation instructions](https://til.simonwillison.net/saelenium/selenium-python-macos)
  - [Windows installation instructions](https://jonathansoma.com/lede/foundations-2018/classes/selenium/selenium-windows-install/)

- LibMagic - An open source tool that identifies file types by checking their headers according to a predefined list of file types.
  - [MacOS installation instructions](https://pypi.org/project/python-magic/#:~:text=python%2Dmagic%2Dbin-,OSX,-When%20using%20Homebrew)
  - [Windows installation instructions](https://pypi.org/project/python-magic/#:~:text=get%20install%20libmagic1-,Windows,-You%27ll%20need%20DLLs)
<br><br>
# Install  

Run the below command in the slangchain project repository parent directory.

```
pip install .
```
<br><br>
# Examples
<br>

## Web Content Document Loader

[UnstructuredURLFileLoader](./slangchain/document_loaders/url_file.py) loads url web content of various types (i.e. html, pdf or json)

<br>

## URL Contextual Compressions Retriever

[UrlCompressedDocSearchChain](./slangchain/chains/url_compressed_doc_search/base.py) loads url web content of various types (i.e. html, pdf or json), and retrives the chunks of text that are most relevant to a search query.

The search uses [LangChain](https://github.com/hwchase17/langchain) [DocumentCompressorPipeline](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/contextual-compression.ipynb). The core idea is simple: given a specific query, we should be able to return only the documents relevant to that query, and only the parts of those documents that are relevant.

The notebook can be found [here](./docs/modules/chains/examples/url_compressed_doc_search.ipynb)