import os
import re
import json
import jieba
import requests
import numpy as np
from loguru import logger
from collections import Counter

from src.llms import GPT
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")
json_response = '''
{\"key_info\": ["新增并网光伏发电容量1060万千瓦", "四分之一", "全国新增光伏电站855万千瓦", "分布式光伏容量205万千瓦", "2014年中国光伏发电量250亿千瓦。", "同比增长超过200%"], 

\"question\": ["2014年中国新增并网光伏发电容量是多少？", "2014年中国新增并网光伏发电容量约占全球新增容量的几分之几？","全国新增光伏电站的容量是多少？", "分布式光伏容量是多少？", "2014年中国光伏发电量是多少？", "2014年中国光伏发电量相比前一年增长了多少？"]}
'''

class QuestEval(GPT):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False, task_name='summary'):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        
        self.quest_gt_save = self._read_quest_gt(f'{task_name}_quest_gt_save.json')

    def save_quest_gt(self, task_name):
        with open(f'src/quest_eval/{task_name}_quest_gt_save.json', 'w', encoding='utf-8') as f:
            json.dump(self.quest_gt_save, f, ensure_ascii=False, indent=4)

    def question_generation(self, text4gen: str):
        prompt = self._read_prompt_template("quest_eval_gen.txt") 
        query = prompt.format(json_response=json_response, news=text4gen)
        
        extracted_content = self.safe_request(query)
        question4eval = json.loads(extracted_content)
        
        return question4eval

    def question_answer(self, context, question):
        template = self._read_prompt_template('quest_eval_answer.txt')
        query = template.format(
            context=context,
            questions=question
        )
        answers = self.safe_request(query)
        
        pattern = r'<response>\n(.*?)\n</response>'
        real_answers = re.findall(pattern, answers, re.DOTALL)
        return real_answers
    
    def _read_prompt_template(self, filename: str):
        path = os.path.join('src/prompts/', filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
        
    def _read_quest_gt(self, filename: str) -> dict:
        path = os.path.join('src/quest_eval', filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.loads(f.read())
        else:
            logger.error(f'Questions generated from ground truth for evaluation not found at {path}')
            return {}
    
    def get_QA_pair(self, data_point: dict):
        ground_truth_text = data_point["ground_truth_text"]
        generated_text = data_point["generated_text"]
        
        if data_point["ID"] in self.quest_gt_save.keys():
            questions_gt = self.quest_gt_save[data_point["ID"]]["question"]
            answers_gt4gt = self.quest_gt_save[data_point["ID"]]["answers"]
        else:
            keyinfo_and_questions = self.question_generation(ground_truth_text)
            questions_gt = keyinfo_and_questions["question"]           
            answers_gt4gt = self.question_answer(ground_truth_text, questions_gt) # 用ground truth回答ground truth生成的问题
            
            keyinfo_and_questions["answers"] = answers_gt4gt
            self.quest_gt_save[data_point["ID"]] = keyinfo_and_questions
    
        answers_gm4gt = self.question_answer(generated_text, questions_gt) # 用generated text回答ground truth生成的问题

        return questions_gt, answers_gt4gt, answers_gm4gt

    def quest_eval(self, data_point: dict):
        try:
            questions_gt, answers_gt4gt, answers_gm4gt = self.get_QA_pair(data_point)

            quest_eval_save = {}
            quest_eval_save["questions_gt"] = questions_gt
            quest_eval_save["answers_gt4gt"] = answers_gt4gt
            quest_eval_save["answers_gm4gt"] = answers_gm4gt

            # 去除ground truth无法推断的问题，说明生成的问题不好，需要排除
            indices = [i for i, x in enumerate(answers_gt4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]

            if len(answers_gm4gt) == 0:
                return 0, 0, quest_eval_save

            undetermined_ratio = answers_gm4gt.count("无法推断") / len(answers_gm4gt)
            quest_recall = 1 - undetermined_ratio

            indices = [i for i, x in enumerate(answers_gm4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]
            
            if answers_gm4gt == []:
                return 0, 0, quest_eval_save

            quest_avg_f1 = word_based_f1_score(answers_gt4gt, answers_gm4gt)

        except Exception as e:
            logger.warning(repr(e))
            quest_eval_save = {}
            quest_eval_save["questions_gt"] = []
            quest_eval_save["answers_gt4gt"] = []
            quest_eval_save["answers_gm4gt"] = []
            return 0, 0, quest_eval_save
        
        return quest_avg_f1, quest_recall, quest_eval_save


def compute_f1(a_gold, a_pred):
    gold_toks = list(jieba.cut(a_gold)) 
    pred_toks = list(jieba.cut(a_pred)) 
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def word_based_f1_score(a_gold_list, a_pred_list):
    f1_list=[]
    for a_gold,a_pred in zip(a_gold_list, a_pred_list):
        f1_list.append(compute_f1(a_gold,a_pred))
    return np.mean(f1_list)

