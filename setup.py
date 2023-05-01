"""setup file"""
import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent


# This call to setup() does all the work
setuptools.setup(
  name="slangchain",
  version="0.0.2",
  description="Slangchain library",
  author="Eugene Tan",
  author_email="bellourchee@gmail.com",
  license="MIT",
  install_requires=[
    'boto3',

    'unstructured',
    'beautifulsoup4',
    'bs4',
    'playwright',
    'selenium',

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

    'faiss-cpu',
    'gptcache',

    'jupyter',
    'langchain==0.0.154'
  ],
  packages=setuptools.find_packages(),
  python_requires=">=3"
)
