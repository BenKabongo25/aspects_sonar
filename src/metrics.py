# Ben Kabongo
# March 2025


import evaluate
import numpy as np
import torch
from typing import List, Dict, Any, Union, Optional, Tuple


def text_evaluation(
    predictions: List[str], 
    references: List[List[str]], 
    device: torch.device, 
    lang: str="en"
) -> Dict[str, Any]:
    
    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    bleu_results["precision"] = np.mean(bleu_results["precisions"])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang=lang, device=device
    )
    bertscore_results["precision"] = np.mean(bertscore_results["precision"])
    bertscore_results["recall"] = np.mean(bertscore_results["recall"])
    bertscore_results["f1"] = np.mean(bertscore_results["f1"])
    del bertscore_results["hashcode"]

    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "n_examples": len(predictions),
        "bertscore": bertscore_results,
        "bleu": bleu_results,
        "rouge": rouge_results,
        "meteor": meteor_results,
    }


def aspect_evaluation(
    predictions: List[List[Tuple[str]]], 
    references: List[List[Tuple[str]]]
) -> Dict[str, Union[int, float]]:
    
    TP = 0
    N_pred = 0
    N_true = 0

    for i in range(len(predictions)):
        pred = set(tuple(sublist) for sublist in predictions[i])
        true = set(tuple(sublist) for sublist in references[i])

        TP += len(pred.intersection(true))
        N_pred += len(pred)
        N_true += len(true)

    precision = TP / N_pred if N_pred > 0 else 0
    recall = TP / N_true if N_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "n_examples": len(predictions),
        "n_pred": N_pred,
        "n_true": N_true,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    