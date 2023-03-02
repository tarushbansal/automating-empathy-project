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


def _predict_empintents(
    test_data: Dict[str, Union[List[str], str]],
    model_args: argparse.Namespace, 
    model: Union[BertForSequenceClassification, RobertaForSequenceClassification], 
    tokenizer: Union[BertTokenizer, RobertaTokenizer], 
    empintent_labels: List[str],
    device: torch.device
) -> None:

    print("Predicting EmpIntents...")
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
        item['empintent_output'] = empintent_labels[int(preds[0])]
        item['empintent_target'] = empintent_labels[int(preds[1])]


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
            item['emotion_output'] = emo_labels[int(preds[0])]
            item['emotion_target'] = emo_labels[int(preds[1])]
            
    return test_data


def _get_epitome_score(
    test_data: Dict[str, Union[List[str], str]], 
    epitome_empathy_scorer: EmpathyScorer
) -> Tuple[List[float]]:
    
    output_IP_scores, output_EX_scores, output_ER_scores = [], [], []
    target_IP_scores, target_EX_scores, target_ER_scores = [], [], []
    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    print("Predicting IP, EX, and ER scores...")
    for item in tqdm(test_data):
        prev_utt = item["context"][-1]
        output = item['output']
        target = item['target']
        
        output_epitome_score = epitome_empathy_scorer([prev_utt], [output])
        target_epitome_score = epitome_empathy_scorer([prev_utt], [target])
        
        item['epitome_IP_output'] = output_epitome_score['IP'][0][0]
        item['epitome_EX_output'] = output_epitome_score['EX'][0][0]
        item['epitome_ER_output'] = output_epitome_score['ER'][0][0]

        item['epitome_IP_target'] = target_epitome_score['IP'][0][0]
        item['epitome_EX_target'] = target_epitome_score['EX'][0][0]
        item['epitome_ER_target'] = target_epitome_score['ER'][0][0]

        output_IP_scores += output_epitome_score['IP'][0]
        output_EX_scores += output_epitome_score['EX'][0]
        output_ER_scores += output_epitome_score['ER'][0]
        
        target_IP_scores += target_epitome_score['IP'][0]
        target_EX_scores += target_epitome_score['EX'][0]
        target_ER_scores += target_epitome_score['ER'][0]

        diff_IP_scores.append(math.pow(abs(output_epitome_score['IP'][0][0] - target_epitome_score['IP'][0][0]), 2))
        diff_EX_scores.append(math.pow(abs(output_epitome_score['EX'][0][0] - target_epitome_score['EX'][0][0]), 2))
        diff_ER_scores.append(math.pow(abs(output_epitome_score['ER'][0][0] - target_epitome_score['ER'][0][0]), 2))
        
    return (
        output_IP_scores, 
        output_EX_scores, 
        output_ER_scores, 
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
        if len(tokens) > args.max_seq_length - special_tokens_count:
            tokens = tokens[:(args.max_seq_length - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]

        # Add [CLS] token
        tokens = [cls_token] + tokens

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence lentargeth.
        padding_lentargeth = args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_lentargeth)
        attention_mask = attention_mask + ([0] * padding_lentargeth)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    # Change to tensor and shift to device
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long).to(device)
    
    return all_input_ids, all_attention_mask


def _get_classifier_args(
    model_dir: str
) -> argparse.Namespace:
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

    if device is None:
        raise ValueError("Must specify device for loading classifier model!")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Specified classifier model directory does not exist!")
        
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
    class_lists: List[str]
) -> Dict[str, float]:
            
    f1_dict = {}
    classwise_metrics = {}
    for class_name in class_lists:
        precisions, recalls, f1s = ConfusionMatrixMetric.compute_metrics(
            pred_labels, 
            target_labels, 
            class_name
        )
        f1_dict[class_name] = f1s
    
        classwise_metrics.update(
            {
                f'{class_name}_precision': float(sum(precisions, None)),
                f'{class_name}_recall': float(sum(recalls, None)),
                f'{class_name}_f1': float(sum(f1s, None))
            }
        )
    
    weighted_f1_score = float(sum(WeightedF1Metric.compute_many(f1_dict), None))

    return weighted_f1_score, classwise_metrics


def _evaluate_acc(preds: List[str], targets: List[str]) -> float:
    preds = np.array(preds)
    targets = np.array(targets)
    acc = float((preds == targets).mean())
    return acc


def compute_empathy_metrics(
    emo_classifier_dir: Optional[str] = None,
    intent_classifier_dir: Optional[str] = None,
    epitome_dir: Optional[str] = None,
    test_data: Optional[Dict] = None, 
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float]]:

    if test_data is None:
        print("USER WARNING: No test data supplied!")
        return {}, {}
    
    # Compute test metrics from the empathy suite
    main_metrics, classwise_metrics = {}, {}

    # Load classifier and predict output emotions
    if emo_classifier_dir is not None:
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
        _predict_emotions(test_data, emo_model_args, emo_model, emo_tokenizer, idx_to_emotions, device)

        emo_outputs = [item['emotion_output'] for item in test_data]
        emo_targets = [item['emotion_target'] for item in test_data]
        main_metrics['emotion_accuracy'] = _evaluate_acc(emo_outputs, emo_targets)
        main_metrics["emotion_f1"], classwise_metrics["emotion"] = _compute_confusion_matrix(
            emo_outputs, 
            emo_targets, 
            EMOTIONS
        )
    
    # Load classifier and predict target and output intents
    if intent_classifier_dir is not None:
        EMPINTENTS = [
            'agreeing', 'acknowledging', 'encouraging', 'consoling', 
            'sympathizing', 'suggesting', 'questioning', 'wishing', 
            'neutral'
        ]
        idx_to_empintents = {i: empintent for i, empintent in enumerate(EMPINTENTS)}
        intent_model_args = _get_classifier_args(intent_classifier_dir)
        intent_tokenizer = _load_classifier_tokenizer(intent_model_args)
        intent_model = _load_classifier_model(intent_classifier_dir, device)
        _predict_empintents(test_data, intent_model_args, intent_model, intent_tokenizer, idx_to_empintents, device)

        empintent_outputs = [item['empintent_output'] for item in test_data]
        empintent_targets = [item['empintent_target'] for item in test_data]
        main_metrics['empintent_accuracy'] = _evaluate_acc(empintent_outputs, empintent_targets)
        main_metrics["empintent_f1"], classwise_metrics["empintent"] = _compute_confusion_matrix(
            empintent_outputs, 
            empintent_targets, 
            EMPINTENTS
        )

    # Load empathy scorer and compute scores
    if epitome_dir is not None:
        epitome_empathy_scorer = EmpathyScorer(epitome_dir, batch_size=1, device=device)
        (output_IP_scores, output_EX_scores, output_ER_scores, 
         target_IP_scores, target_EX_scores, target_ER_scores, 
         diff_IP_scores, diff_EX_scores, diff_ER_scores) = _get_epitome_score(test_data, epitome_empathy_scorer)
        
        main_metrics['epitome_IP_output'] = sum(output_IP_scores) / len(output_IP_scores)
        main_metrics['epitome_EX_output'] = sum(output_EX_scores) / len(output_EX_scores)
        main_metrics['epitome_ER_output'] = sum(output_ER_scores) / len(output_ER_scores)

        main_metrics['epitome_IP_target'] = sum(target_IP_scores) / len(target_IP_scores)
        main_metrics['epitome_EX_target'] = sum(target_EX_scores) / len(target_EX_scores)
        main_metrics['epitome_ER_target'] = sum(target_ER_scores) / len(target_ER_scores)

        main_metrics['diff_IP'] = sum(diff_IP_scores) / len(diff_IP_scores)
        main_metrics['diff_EX'] = sum(diff_EX_scores) / len(diff_EX_scores)
        main_metrics['diff_ER'] = sum(diff_ER_scores) / len(diff_ER_scores)

    return main_metrics, classwise_metrics

# --------------------------------------------------------------------------------------
