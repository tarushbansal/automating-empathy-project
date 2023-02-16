# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import pairwise_cosine_similarity
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist

# ------------------------- IMPLEMENTATION -----------------------------------


def compute_test_metrics(
    targets: List[str],
    predictions: List[str],
    encoded_targets: Optional[List[List[int]]] = None,
    encoded_predictions: Optional[List[List[int]]] = None,
    word_embeddings: Optional[nn.Module] = None,
    metric_n_grams: int = 4,
) -> Dict[str, float]:

    count_n_grams = [0 for _ in range(metric_n_grams)]
    unique_n_grams = [set() for _ in range(metric_n_grams)]
    avg_bow, extrema_bow, greedy_bow = [], [], []

    for i in range(len(targets)):
        targets[i] = [targets[i].strip().split(" ")]
        predictions[i] = predictions[i].strip().split(" ")

        # DIST calculation
        for n in range(metric_n_grams):
            count = len(predictions[i]) - n
            count_n_grams[n] += count
            unique_n_grams[n].update([tuple(predictions[i][j:j+n+1])
                                      for j in range(count)])

        if word_embeddings is not None:
            # Embed both targets and predictions
            device = word_embeddings.weight.device
            target_embeddings = word_embeddings(
                torch.LongTensor(encoded_targets[i:i+1]).to(device))
            pred_embeddings = word_embeddings(
                torch.LongTensor(encoded_predictions[i:i+1]).to(device))

            # Average BOW
            avg_target_embed = target_embeddings.mean(dim=1)
            avg_pred_embed = pred_embeddings.mean(dim=1)
            avg_bow.append(
                float(F.cosine_similarity(avg_target_embed, avg_pred_embed)))

            # Extrema BOW
            max_target, _ = torch.max(target_embeddings, dim=1)
            min_target, _ = torch.min(target_embeddings, dim=1)
            mask_target = (torch.abs(max_target) >= torch.abs(min_target))
            extrema_target_embed = torch.where(mask_target, max_target, min_target)
            max_pred, _ = torch.max(pred_embeddings, dim=1)
            min_pred, _ = torch.min(pred_embeddings, dim=1)
            mask_pred = (torch.abs(max_pred) >= torch.abs(min_pred))
            extrema_pred_embed = torch.where(mask_pred, max_pred, min_pred)
            extrema_bow.append(
                float(F.cosine_similarity(extrema_target_embed, extrema_pred_embed)))

            # Greedy BOW
            sim = pairwise_cosine_similarity(
                target_embeddings.squeeze(dim=0),
                pred_embeddings.squeeze(dim=0)
            )
            greedy_bow.append(
                float(sim.max(dim=0)[0].mean() + sim.max(dim=1)[0].mean() / 2))

    test_metrics = {
        "bleu": corpus_bleu(targets, predictions, weights=[1/metric_n_grams]*metric_n_grams),
        "nist": corpus_nist(targets, predictions, n=metric_n_grams),
    }

    for n in range(metric_n_grams):
        test_metrics[f"dist-{n+1}"] = len(unique_n_grams[n]) / count_n_grams[n]

    if word_embeddings is not None:
        test_metrics["avg_bow"] = sum(avg_bow) / len(avg_bow)
        test_metrics["extrema_bow"] = sum(extrema_bow) / len(extrema_bow)
        test_metrics["greedy_bow"] = sum(greedy_bow) / len(greedy_bow)

    return test_metrics

# ------------------------------------------------------------------------------------
