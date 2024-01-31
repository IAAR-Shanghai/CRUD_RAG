import os
import re
import datetime
from abc import ABC
from loguru import logger
from src.metric.common import (
    bleu_score, 
    rougeL_score, 
    bert_score,
)
from src.metric.quest_eval import QuestEval

class BaseTask(ABC):
    def __init__(
            self,
            output_dir: str = './output',
            quest_eval_model: str = "gpt-3.5-turbo",
            use_quest_eval: bool = False,
            use_bert_score: bool = False,
        ):
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        
        self.use_quest_eval = use_quest_eval
        self.use_bert_score = use_bert_score
        if self.use_quest_eval: 
            self.quest_eval = QuestEval(
                model_name=quest_eval_model, temperature=0.1, 
                max_new_tokens=1280, task_name=self.__class__.__name__
            )
        
    
    def set_model(self, model, retriever) -> None:
        
        return 
    
    def retrieve_docs(self, obj:dict) -> str:

        return " "

    def model_generation(self, obj:dict) -> None:
        # use LLM to generate text
        
        return     
        
    def _read_prompt_template(self, filename: str):
        # read template to generate prompt
        
        return

    def scoring(self, data_point: dict) -> dict:
        return {
            'metrics': {
                # Numerical results to be recorded by subclasses, mandatory.
                # Such as accuracy, recall, bleu, rouge, etc.
            },
            'log': {
                # String results to be recorded by subclasses, optional.
                # Such as model output.
            },
            'valid': ...
                # Boolean result to be recorded by subclasses, indicating whether the evaluation is valid, mandatory.
                # True or False.
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
                # 'Metric1': Value,
                # 'Metric2': Value,
                # ...
        }

    