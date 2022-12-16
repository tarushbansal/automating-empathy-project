# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import math
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
    encoded_targets: List[List[int]],
    encoded_predictions: List[List[int]],
    pred_n_grams: int = 4,
    log_prob: Optional[float] = None,
    word_embeddings: Optional[nn.Module] = None,
) -> Dict[str, float]:

    n_unigrams, n_bigrams = 0, 0
    unique_unigrams, unique_bigrams = set(), set()
    avg_bow, extrema_bow, greedy_bow = [], [], []

    for i in range(len(targets)):
        targets[i] = [targets[i].split(" ")]
        predictions[i] = predictions[i].split(" ")

        # DIST - unigrams and bigrams
        unique_unigrams.update([tuple(predictions[i][j:j+1])
                                for j in range(len(predictions[i]))])
        unique_bigrams.update([tuple(predictions[i][j:j+2])
                               for j in range(len(predictions[i]) - 1)])
        n_unigrams += len(predictions[i])
        n_bigrams += len(predictions[i]) - 1

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
        "bleu": corpus_bleu(targets, predictions, weights=[1/pred_n_grams]*pred_n_grams),
        "nist": corpus_nist(targets, predictions, n=pred_n_grams),
        "dist-1": len(unique_unigrams) / n_unigrams,
        "dist-2": len(unique_bigrams) / n_bigrams
    }

    if log_prob is not None:
        num_tokens = sum([len(enc) - 1 for enc in encoded_predictions])
        test_metrics["ppl"] = math.exp(-log_prob/num_tokens)
    if word_embeddings is not None:
        test_metrics["avg_bow"] = sum(avg_bow) / len(avg_bow)
        test_metrics["extrema_bow"] = sum(extrema_bow) / len(extrema_bow)
        test_metrics["greedy_bow"] = sum(greedy_bow) / len(greedy_bow)

    return test_metrics

# ------------------------------------------------------------------------------------
