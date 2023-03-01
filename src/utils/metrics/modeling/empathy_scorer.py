# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import numpy as np

import torch
import torch.nn as nn
from transformers import RobertaTokenizer

# User-Defined Modules
from utils.metrics.modeling.models import  BiEncoderAttentionWithRationaleClassification

# -------------------------- IMPLEMENTATION -----------------------------------


class EmpathyScorer(nn.Module):
    def __init__(
        self, 
        dir: str, 
        batch_size: int, 
        device: torch.device
    ) -> None:
        super().__init__()

        if device is None:
            raise ValueError("Must specify device for loading epitome models!")

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.batch_size = batch_size

        self.model_IP = BiEncoderAttentionWithRationaleClassification()
        self.model_EX = BiEncoderAttentionWithRationaleClassification()
        self.model_ER = BiEncoderAttentionWithRationaleClassification()

        print(f"Loading IP model from '{dir}'")
        IP_weights = torch.load(os.path.join(dir, 'finetuned_IP.pth'), map_location=device)
        self.model_IP.load_state_dict(IP_weights)
        self.model_IP.to(device)

        print(f"Loading EX model from '{dir}'")
        EX_weights = torch.load(os.path.join(dir, 'finetuned_EX.pth'), map_location=device)
        self.model_EX.load_state_dict(EX_weights)
        self.model_EX.to(device)
        
        print(f"Loading ER model from '{dir}'")
        ER_weights = torch.load(os.path.join(dir, 'finetuned_ER.pth'), map_location=device)
        self.model_ER.load_state_dict(ER_weights)
        self.model_ER.to(device)

        self.device = device


    def forward(self, seeker_post, response_post):
        self.model_IP.eval()
        self.model_EX.eval()
        self.model_ER.eval()

        seeker_input = self.tokenizer.batch_encode_plus(
            seeker_post,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 64,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            padding=True,
        )
        response_input = self.tokenizer.batch_encode_plus(
            response_post,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 64,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            padding=True
        )

        seeker_input['input_ids'] = seeker_input['input_ids'].to(self.device)
        seeker_input['attention_mask'] = seeker_input['attention_mask'].to(self.device)
        response_input['input_ids'] = response_input['input_ids'].to(self.device)
        response_input['attention_mask'] = response_input['attention_mask'].to(self.device)
            
        with torch.no_grad():
            (logits_empathy_IP, logits_rationale_IP,) = self.model_IP(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )
            (logits_empathy_EX, logits_rationale_EX,) = self.model_EX(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )
            (logits_empathy_ER, logits_rationale_ER,) = self.model_ER(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )

        logits_empathy_IP = torch.nn.functional.softmax(logits_empathy_IP, dim=1)
        logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy()
        logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
        empathy_predictions_IP = np.argmax(logits_empathy_IP, axis=1).tolist()
        rationale_predictions_IP = np.argmax(logits_rationale_IP, axis=2)

        logits_empathy_EX = torch.nn.functional.softmax(logits_empathy_EX, dim=1)
        logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy()
        logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
        empathy_predictions_EX = np.argmax(logits_empathy_EX, axis=1).tolist()
        rationale_predictions_EX = np.argmax(logits_rationale_EX, axis=2)
        
        logits_empathy_ER = torch.nn.functional.softmax(logits_empathy_ER, dim=1)
        logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy()
        logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
        empathy_predictions_ER = np.argmax(logits_empathy_ER, axis=1).tolist()
        rationale_predictions_ER = np.argmax(logits_rationale_ER, axis=2)
        
        return {
            'IP': (empathy_predictions_IP, logits_empathy_IP, rationale_predictions_IP),
            'EX': (empathy_predictions_EX, logits_empathy_EX, rationale_predictions_EX),
            'ER': (empathy_predictions_ER, logits_empathy_ER, rationale_predictions_ER)
        }

# --------------------------------------------------------------------------------------
