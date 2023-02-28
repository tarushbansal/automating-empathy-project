# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Tuple

import torch

from transformers import AutoModelForSequenceClassification
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    RobertaForSequenceClassification, 
    RobertaTokenizer
)

from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric

# User-Defined Modules
from utils.metrics.modeling.empathy_scorer import EmpathyScorer

# -------------------------- IMPLEMENTATION -----------------------------------


def _pred_empintents(
    test_data: Dict[str, Union[List[str], str]],
    model_args: argparse.Namespace, 
    model: Union[BertForSequenceClassification, RobertaForSequenceClassification], 
    tokenizer: Union[BertTokenizer, RobertaTokenizer], 
    empintent_labels: List[str],
    device: torch.device
) -> None:

    print("Predicting EmpIntents...")
    for item in tqdm(test_data):
        context = tokenizer.sep_token.join(item['context'])
        output = item['output']
        target = item['target']
        
        input_data = [context, output, target]
        input_ids, attention_mask = _convert_data_to_tensors(
            input_data, 
            model_args, 
            tokenizer,
            device
        )

        with torch.no_grad():
            input_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None
            }
            outputs = model(**input_kwargs)
            logits = outputs[0]

            logits = torch.nn.functional.softmax(logits, dim=1)

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
        
        assert len(preds) == 3
        item['empintent-context'] = empintent_labels[int(preds[0])]
        item['empintent-pred'] = empintent_labels[int(preds[1])]
        item['empintent-target'] = empintent_labels[int(preds[2])]


def _predict_emotions(
    test_data: Dict[str, Union[List[str], str]],
    model_args, 
    model: Union[BertForSequenceClassification, RobertaForSequenceClassification], 
    tokenizer: Union[BertTokenizer, RobertaTokenizer], 
    emo_labels: List[str],
    device: torch.device
) -> None:

    print("Predicting Emotions...")
    for item in tqdm(test_data):
        output = item['output']
        target = item['target']

        input_data = [output, target]
        input_ids, attention_mask = _convert_data_to_tensors(
            input_data, 
            model_args, 
            tokenizer,
            device
        )

        with torch.no_grad():
            input_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None
            }
            outputs = model(**input_kwargs)
            logits = outputs[0]

            logits = torch.nn.functional.softmax(logits, dim=1)

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            assert len(preds) == 2
            item['emotion-pred'] = emo_labels[int(preds[0])]
            item['emotion-target'] = emo_labels[int(preds[1])]
            
    return test_data


def _get_epitome_score(
    test_data: Dict[str, Union[List[str], str]], 
    epitome_empathy_scorer: EmpathyScorer
) -> Tuple[List[float]]:
    
    pred_IP_scores, pred_EX_scores, pred_ER_scores = [], [], []
    target_IP_scores, target_EX_scores, target_ER_scores = [], [], []
    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    print("Predicting IP, EX, and ER scores...")
    for item in tqdm(test_data):
        context = epitome_empathy_scorer.tokenizer.sep_token.join(item['context'])
        output = item['output']
        target = item['target']
        
        pred_epitome_score = epitome_empathy_scorer([context], [output])
        tar_get_epitome_score = epitome_empathy_scorer([context], [target])
        
        item['epitome-IP-pred'] = int(pred_epitome_score['IP'][0][0])
        item['epitome-EX-pred'] = int(pred_epitome_score['EX'][0][0])
        item['epitome-ER-pred'] = int(pred_epitome_score['ER'][0][0])

        item['epitome-IP-target'] = int(tar_get_epitome_score['IP'][0][0])
        item['epitome-EX-target'] = int(tar_get_epitome_score['EX'][0][0])
        item['epitome-ER-target'] = int(tar_get_epitome_score['ER'][0][0])

        pred_IP_scores += pred_epitome_score['IP'][0]
        pred_EX_scores += pred_epitome_score['EX'][0]
        pred_ER_scores += pred_epitome_score['ER'][0]
        
        target_IP_scores += tar_get_epitome_score['IP'][0]
        target_EX_scores += tar_get_epitome_score['EX'][0]
        target_ER_scores += tar_get_epitome_score['ER'][0]

        diff_IP_scores.append(math.pow(abs(pred_epitome_score['IP'][0][0] - tar_get_epitome_score['IP'][0][0]), 2))
        diff_EX_scores.append(math.pow(abs(pred_epitome_score['EX'][0][0] - tar_get_epitome_score['EX'][0][0]), 2))
        diff_ER_scores.append(math.pow(abs(pred_epitome_score['ER'][0][0] - tar_get_epitome_score['ER'][0][0]), 2))
        
    return (
        pred_IP_scores, 
        pred_EX_scores, 
        pred_ER_scores, 
        target_IP_scores, 
        target_EX_scores, 
        target_ER_scores, 
        diff_IP_scores, 
        diff_EX_scores, 
        diff_ER_scores
    )


def _convert_data_to_tensors(
    utterances: List[str],
    args: argparse.Namespace,
    tokenizer: Union[BertTokenizer, RobertaTokenizer],
    device: torch.device
) -> Tuple[torch.Tensor]:
    
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids, all_attention_mask = [], []
    
    for utt in utterances:
        tokens = tokenizer.tokenize(utt)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_lentargeth - special_tokens_count:
            tokens = tokens[:(args.max_seq_lentargeth - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]

        # Add [CLS] token
        tokens = [cls_token] + tokens

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence lentargeth.
        padding_lentargeth = args.max_seq_lentargeth - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_lentargeth)
        attention_mask = attention_mask + ([0] * padding_lentargeth)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    # Change to tensor and shift to device
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long).to(device)
    
    return all_input_ids, all_attention_mask


def _get_classifier_args(model_dir: str) -> argparse.Namespace:

    print(f"Loaded classifier args from '{model_dir}'")
    return torch.load(os.path.join(model_dir, 'training_args.bin'))


def _load_classifier_tokenizer(
    args: argparse.Namespace, 
    add_prefix_space: bool = False
) -> Union[BertTokenizer, RobertaTokenizer]:
    
    print(f"Loaded classifier tokenizer")
    if args.model_type == "bert":
        return BertTokenizer.from_pretrained(
            args.model_name_or_path, 
            add_prefix_space=add_prefix_space
        )
    elif args.model_type == "roberta":
        return RobertaTokenizer.from_pretrained(
            args.model_name_or_path, 
            add_prefix_space=add_prefix_space
        )
    else:
        ValueError("Model type not supported!")


def _load_classifier_model(
    model_dir: str, 
    device: torch.device
) -> Union[BertForSequenceClassification, RobertaForSequenceClassification]:

    if not os.path.exists(model_dir):
        raise FileNotFoundError("Model doesn't exists! Train first!")
        
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        print(f"Loaded classifier model from '{model_dir}'")
        return model
    except:
        raise FileNotFoundError("Some model files might be missing...")    


def _compute_confusion_matrix(
    pred_labels: List[str], 
    target_labels: List[str], 
    class_lists: List[str], 
    type: str
) -> Dict[str, float]:
            
    f1_dict = {}
    emo_metrics = {}
    for class_name in class_lists:
        precisions, recalls, f1s = ConfusionMatrixMetric.compute_metrics(
            pred_labels, 
            target_labels, 
            class_name
        )
        f1_dict[class_name] = f1s
    
        emo_metrics.update(
            {
                f'{class_name}_precision': sum(precisions, None),
                f'{class_name}_recall': sum(recalls, None),
                f'{class_name}_f1': sum(f1s, None)
            }
        )
    
    emo_metrics[f'{type}-f1'] = sum(WeightedF1Metric.compute_many(f1_dict), None)

    return emo_metrics


def _evaluate_acc(preds: List[str], targets: List[str]) -> float:
    preds = np.array(preds)
    targets = np.array(targets)
    acc = (preds == targets).mean()
    return acc


def compute_empathy_metrics(
    test_data: Dict[str, Union[List[str], str]], 
    device: torch.device,
    emo_classifier_dir: Optional[str] = None,
    intent_classifier_dir: Optional[str] = None,
    epitome_model_dir: Optional[str] = None
) -> Dict[str, float]:

    # Compute test metrics from the empathy suite
    emo_metrics = {}
    
    # Load classifier and predict output emotions
    if emo_classifier_dir is not None and os.path.isdir(emo_classifier_dir):
        EMOTIONS = [
        'proud', 'apprehensive', 'disappointed', 'faithful', 
        'impressed', 'devastated', 'prepared', 'nostalgic', 'annoyed', 
        'grateful', 'joyful', 'terrified', 'caring', 'trusting', 'sad', 
        'guilty', 'sentimental', 'hopeful', 'confident', 'surprised', 
        'furious', 'afraid', 'jealous', 'excited', 'lonely', 'disgusted', 
        'embarrassed', 'angry', 'content', 'ashamed', 'anticipating', 'anxious'
        ]
        idx_to_emotions = {i: emo for i, emo in enumerate(EMOTIONS)}
        emo_model_args = _get_classifier_args(emo_classifier_dir)
        emo_tokenizer = _load_classifier_tokenizer(emo_model_args)
        emo_model = _load_classifier_model(emo_classifier_dir, device)
        _predict_emotions(test_data, emo_model_args, emo_tokenizer, emo_model, idx_to_emotions, device)

        emo_preds = [item['emotion_pred'] for item in test_data]
        emo_targets = [item['emotion_target'] for item in test_data]
        emo_metrics['emo_accuracy'] = _evaluate_acc(emo_preds, emo_targets)
        emo_metrics.update(_compute_confusion_matrix(emo_preds, emo_targets, EMOTIONS, 'emotion'))
    
    # Load classifier and predict target and output intents
    if intent_classifier_dir is not None and os.path.isdir(intent_classifier_dir):
        EMPINTENTS = [
            'agreeing', 'acknowledging', 'encouraging', 'consoling', 
            'sympathizing', 'suggesting', 'questioning', 'wishing', 
            'neutral'
        ]
        idx_to_empintents = {i: empintent for i, empintent in enumerate(EMPINTENTS)}
        intent_model_args = _get_classifier_args(intent_classifier_dir)
        intent_tokenizer = _load_classifier_tokenizer(intent_model_args)
        intent_model = _load_classifier_model(intent_classifier_dir, device)
        _pred_empintents(test_data, intent_model_args, intent_tokenizer, intent_model, idx_to_empintents, device)

        empintent_preds = [item['empintent_pred'] for item in test_data]
        empintent_targets = [item['empintent_target'] for item in test_data]
        emo_metrics['empintent_accuracy'] = _evaluate_acc(empintent_preds, empintent_targets)
        emo_metrics.update(_compute_confusion_matrix(empintent_preds, empintent_targets, EMPINTENTS, 'empintent'))

    # Load empathy scorer and compute scores
    if epitome_model_dir is not None and os.path.isdir(epitome_model_dir):
        epitome_empathy_scorer = EmpathyScorer(epitome_model_dir, batch_size=1, device=device)
        (pred_IP_scores, pred_EX_scores, pred_ER_scores, 
         target_IP_scores, target_EX_scores, target_ER_scores, 
         diff_IP_scores, diff_EX_scores, diff_ER_scores) = _get_epitome_score(test_data, epitome_empathy_scorer)
        
        emo_metrics['epitome_IP_pred'] = sum(pred_IP_scores) / len(pred_IP_scores)
        emo_metrics['epitome_EX_pred'] = sum(pred_EX_scores) / len(pred_EX_scores)
        emo_metrics['epitome_ER_pred'] = sum(pred_ER_scores) / len(pred_ER_scores)

        emo_metrics['epitome_IP_pred'] = sum(target_IP_scores) / len(target_IP_scores)
        emo_metrics['epitome_EX_pred'] = sum(target_EX_scores) / len(target_EX_scores)
        emo_metrics['epitome_ER_pred'] = sum(target_ER_scores) / len(target_ER_scores)

        emo_metrics['diff_IP'] = sum(diff_IP_scores) / len(diff_IP_scores)
        emo_metrics['diff_EX'] = sum(diff_EX_scores) / len(diff_EX_scores)
        emo_metrics['diff_ER'] = sum(diff_ER_scores) / len(diff_ER_scores)

    return emo_metrics

# --------------------------------------------------------------------------------------
