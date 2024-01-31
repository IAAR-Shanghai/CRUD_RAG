# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


from typing import Callable

import evaluate
import jieba
from loguru import logger
from text2vec import Similarity


def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(repr(e))
    return wrapper


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    with_penalty = False
) -> float:
    f = lambda text: list(jieba.cut(text))
    bleu = evaluate.load('src/.cache/huggingface/bleu')
    results = bleu.compute(predictions=[continuation], references=[[reference]], tokenizer=f)
    
    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty==0 else bleu_avg/brevity_penalty, bleu1, bleu2, bleu3, bleu4


@catch_all_exceptions
def rougeL_score(
    continuation: str,
    reference: str
) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load('src/.cache/huggingface/rouge')
    results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score


@catch_all_exceptions
def kw_precision(
    continuation: str,
    reference: str,
    kw_extracter: Callable[[str], list[str]],
    with_kw_list: bool = True
) -> float | tuple[float, list[str], list[str]]:
    """Measure the rationality of a generated continuation sentence with respect to the original news object."""
    kws = kw_extracter(continuation)
    if len(kws) == 0:
        return 0, [], [] if with_kw_list else 0
    appeared_kws = [kw for kw in kws if kw in reference]
    precision = len(appeared_kws) / len(kws)
    return precision, appeared_kws, kws if with_kw_list else precision


@catch_all_exceptions
def bert_score(
    continuation: str,
    reference: str
) -> float:
    """
    Note:
        Requesting the network to connect to Hugging Face. 
    """
    sim = Similarity(model_name_or_path="src/.cache/text2vec-base-chinese")
    score = sim.get_score(continuation, reference)
    return score


def classifications(
    predictions: list[bool],
    references: list[bool]
) -> tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 in a binary classification problem.

    Args:
        predictions (list[bool]): List of predicted values (0 or 1).
        references (list[bool]): List of true values (0 or 1).

    Returns:
        tuple: Accuracy, precision, recall, and F1 scores.

    """
    true_positive = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 1)
    false_positive = sum(1 for a, b in zip(references, predictions) if a == 0 and b == 1)
    false_negative = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = sum(1 for a, b in zip(references, predictions) if a == b) / len(predictions) if len(predictions) > 0 else 0
    return accuracy, precision, recall, f1
