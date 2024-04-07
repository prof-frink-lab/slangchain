"""setup file"""
import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent


# This call to setup() does all the work
setuptools.setup(
  name="slangchain",
  version="0.0.4",
  description="Slangchain library",
  author="Eugene Tan",
  author_email="bellourchee@gmail.com",
  license="MIT",
  install_requires=[
    'validators==0.19.0',

    'boto3',

    'unstructured',
    'beautifulsoup4',
    'bs4',
    'playwright',
    'selenium',
    'lark',

    'huggingface-hub',
    'nltk',
    'nomic',
    'openai',
    'sentence-transformers',
    'spacy',
    'tiktoken',
    'torch',
    'tokenizers',
    'transformers',

    'python-magic',
    'libmagic',
    'pdfminer.six',
    'poppler-utils',
    'pymupdf',
    'pypdf',
    'pytesseract',

    'faiss-cpu',
    'chromadb',
    'gptcache',
    'pinecone-client',

    'jupyter',
    'pydantic==1.10.9',
    'langchain==0.1.14',
    'langchain_experimental==0.0.53',
    'langchain_googledrive==0.1.14',
    'langchain_openai==0.0.8',
    'langchain_anthropic==0.1.6',
    'langgraph==0.0.24',
    'langchainhub==0.1.15'
  ],
  packages=setuptools.find_packages(),
  package_data={'': ['mark_page.js']},
  python_requires=">=3"
)
