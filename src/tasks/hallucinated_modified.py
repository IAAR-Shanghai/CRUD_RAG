import os
import re
import datetime
from src.tasks.base import BaseTask
from loguru import logger
from src.metric.common import (
    bleu_score, 
    rougeL_score, 
    bert_score,
)
from src.metric.quest_eval import QuestEval


class HalluModified(BaseTask):
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
        self.model = model
        self.retriever = retriever
    
    def retrieve_docs(self, obj:dict) -> str:
        query_text = obj["newsBeginning"]
        retrieve_context = self.retriever.search_docs(query_text)
        retrieve_context = retrieve_context.split('\nGiven the context information')[0]
        return retrieve_context

    def model_generation(self, obj:dict):
        if obj["hallucinatedMod"] == '","msg":"request openai failed"':
            return '","msg":"request openai failed"'
        template = self._read_prompt_template('hallu_mod.txt')
        query = template.format(
            begin=f'{obj["newsBeginning"]}',
            hallu_continue=f'{obj["hallucinatedContinuation"]}',
            search_documents=f'{obj["retrieve_context"]}'
        )
        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        return real_res.strip()

    def _read_prompt_template(self, filename: str):
        path = os.path.join('src/prompts/', filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    def scoring(self, data_point: dict) -> dict:
        generated_text = data_point["generated_text"]
        ground_truth_text = data_point["hallucinatedMod"]
        data_point["ground_truth_text"] = ground_truth_text
           
        if self.use_quest_eval:
            QA_avg_F1, QA_recall, quest_eval_save = self.quest_eval.quest_eval(data_point)
        else:
            QA_avg_F1, QA_recall, quest_eval_save = 0.0, 0.0, {}
        
        if self.use_bert_score:
            bertscore = bert_score(generated_text, ground_truth_text)
        else:
            bertscore = 0.0
        
        bleu_avg, bleu1, bleu2, bleu3, bleu4 = bleu_score(generated_text, ground_truth_text)

        return {
            'metrics': {
                'bleu-avg': bleu_avg or 0.0,
                'bleu-1': bleu1 or 0.0,
                'bleu-2': bleu2 or 0.0,
                'bleu-3': bleu3 or 0.0,
                'bleu-4': bleu4 or 0.0,
                'rouge-L': rougeL_score(generated_text, ground_truth_text) or 0.0,
                'bertScore': bertscore,
                'QA_avg_F1': QA_avg_F1,
                'QA_recall': QA_recall,
                'length': len(generated_text)
            },
            'log': {
                'generated_text': generated_text,
                'ground_truth_text': ground_truth_text,
                'quest_eval_save': quest_eval_save,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': len(generated_text.strip()) != 0
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'bleu-avg': 0, 'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 
                   'bleu-4': 0, 'rouge-L': 0, 'bertScore': 0, 'QA_avg_F1': 0, 
                   'QA_recall': 0, 'length': 0}
        
        valid_qa_count = 0
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
            if self.use_quest_eval and result['log']['quest_eval_save']['questions_gt'] != []:
                valid_qa_count += 1
        
        overall_save = {f'avg. {key}': value / len(results) for key, value in overall.items() if key != 'QA_avg_F1' and key != 'QA_recall'}
        if self.use_quest_eval:
            overall_save['QA_avg_F1'] = overall['QA_avg_F1'] / valid_qa_count
            overall_save['QA_recall'] = overall['QA_recall'] / valid_qa_count
        overall_save['num'] = len(results)
       
        return overall_save

