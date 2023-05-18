import sys
import os
import logging

from typing import Dict, List, Any, Optional

from transformers import (
  AutoTokenizer,
  AutoModelForTokenClassification,
  TokenClassificationPipeline)

from transformers.pipelines import AggregationStrategy

from pydantic import BaseModel, Field

LABEL_KEY = "label"
TEXT_KEY = "text"
START_CHAR_KEY = "start_char"
END_CHAR_KEY = "end_char"



class KeyPhraseExtractor():
  """NER Phrase Token Classifier"""

  label_key : str =  Field(default="key_phrase_ner")
  model_name : str = Field(default="ml6team/keyphrase-extraction-distilbert-inspec")
  model : AutoModelForTokenClassification = Field(default=None)
  predictions : List[str] = Field(default=[])

  def __init__(
    self,
    *args,
    model_name:str="ml6team/keyphrase-extraction-distilbert-inspec",
    **kwargs):

    self.model = KeyPhraseExtractorPipeline(model=model_name, *args, **kwargs)

  def inference(self, sentence: str) -> Dict[str, Any]:
    """inference"""
    self.predictions = self.model(sentence)

    return self.predictions

class KeyPhraseExtractorPipeline(TokenClassificationPipeline):
  """Key phrase extractor pipeline class"""
  def __init__(self, model, *args, **kwargs):
    super().__init__(
      model=AutoModelForTokenClassification.from_pretrained(model),
      tokenizer=AutoTokenizer.from_pretrained(model),
      *args,
      **kwargs)

  def postprocess(
    self,
    all_outputs,
    aggregation_strategy=AggregationStrategy.FIRST,
    ignore_labels=None):
    """Post process function"""

    results = super().postprocess(
      all_outputs=all_outputs,
      aggregation_strategy=aggregation_strategy,
      ignore_labels=ignore_labels)

    return list({result.get("word").strip().lower() for result in results})
