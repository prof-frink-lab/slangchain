"""NER Entity Token Classifier"""
from typing import Dict, List, Any
from pydantic import Field
from transformers import (
  pipeline,
  AutoTokenizer,
  AutoModelForTokenClassification)

class EntityExtractor():
  """Entity Extractor"""

  _input_text: str = Field(default=None)
  _predictions: List[Dict[str, Any]] = Field(default=[])
  _locations: List[str] = Field(default=[])
  _organsiations: List[str] = Field(default=[])
  _persons: List[str] = Field(default=[])

  def __init__(
    self,
    model_name:str="Jean-Baptiste/roberta-large-ner-english"):


    self._model_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(model_name)

    self._model = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

  def inference(self, input_text:str) -> Dict:
    """inference function"""

    self._input_text = input_text

    self._predictions = self._model(self._input_text)

    return self._predictions

  @property
  def persons(self) -> List[str]:
    """Get persons function"""
    if not self._predictions:
      self._predictions = self.inference(self._input_text)

    self._persons = [ i["word"].strip().lower() for i in self._predictions \
      if i["entity_group"] == "PER"]

    return self._persons

  @property
  def organisations(self) -> List[str]:
    """Get organisations function"""
    if not self._predictions:
      self._predictions = self.inference(self._input_text)

    self._organsiations = [ i["word"].strip().lower() \
      for i in self._predictions \
      if i["entity_group"] == "ORG"]

    return self._organsiations

  @property
  def locations(self) -> List[str]:
    """Get locations function"""
    if not self._predictions:
      self._predictions = self.inference(self._input_text)

    self._locations = [ i["word"].strip().lower() for i in self._predictions \
      if i["entity_group"] == "LOC"]

    return self._locations
