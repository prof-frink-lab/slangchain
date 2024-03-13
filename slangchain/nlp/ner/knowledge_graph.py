"""Knowledge Graph"""
from typing import Dict, List, Any
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer)
from transformers.pipelines.text2text_generation import ReturnType
from langchain_core.pydantic_v1 import Field

class KnowledgeGraph():
  """Knowledge Graph class"""

  max_new_tokens: int = Field(
    default=200, description="Maximum number of new tokens to generate."
  )
  predictions : List[Dict[str, Any]] = Field(default=[])

  def __init__(
    self,
    *args,
    model_name:str='Babelscape/rebel-large',
    **kwargs):

    self.model = EntityRelationshipExtractionPipeline(model=model_name, *args, **kwargs)

  def inference(self, sentence: str) -> Dict[str, Any]:
    """inference"""
    self.predictions = self.model(sentence)

    return self.predictions

class EntityRelationshipExtractionPipeline(Text2TextGenerationPipeline):
  """Key phrase extractor pipeline class"""
  def __init__(self, model, *args, **kwargs):
    super().__init__(
      model=AutoModelForSeq2SeqLM.from_pretrained(model),
      tokenizer=AutoTokenizer.from_pretrained(model),
      *args,
      **kwargs)

  def _extract_entity_relationships(self, text) -> List[Dict[str, Any]]:
    triplets = []
    relation = ""
    subject = ""
    relation = ""
    object_ = ""
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
      if token == "<triplet>":
        current = 't'
        if relation != '':
          triplets.append(
            { "subject": subject.strip(), "relation": relation.strip(), "object": object_.strip() })
          relation = ''
        subject = ''
      elif token == "<subj>":
        current = 's'
        if relation != '':
          triplets.append(
            { "subject": subject.strip(), "relation": relation.strip(), "object": object_.strip() })
        object_ = ''
      elif token == "<obj>":
        current = 'o'
        relation = ''
      else:
        if current == 't':
          subject += ' ' + token
        elif current == 's':
          object_ += ' ' + token
        elif current == 'o':
          relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
      triplets.append(
        { "subject": subject.strip(), 'relation': relation.strip(), 'object': object_.strip() })
    return triplets

  def postprocess(
    self,
    model_outputs,
    return_type=ReturnType.TEXT,
    clean_up_tokenization_spaces=False) -> List[Dict[str, Any]]:
    """Post process function"""

    records = []
    for output_ids in model_outputs["output_ids"][0]:
      if return_type == ReturnType.TEXT:
        result = self.tokenizer.decode(
          output_ids,
          skip_special_tokens=False,
          clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        results = self._extract_entity_relationships(result)
        records += results
    return records
