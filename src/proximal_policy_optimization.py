# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import copy
from typing import List, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# User-defined Modules
from dialogue_model_supervisor import DialogueModelSupervisor
from reward_model_supervisor import RewardModelSupervisor
from data_classes import (
    DecoderModelBatch, 
    EncoderDecoderModelBatch, 
    RewardModelBatch, 
    PPOConfig
)

# ------------------------- IMPLEMENTATION -----------------------------------


class PPOSupervisor(pl.LightningModule):
    def __init__(
        self,
        dialogue_model: DialogueModelSupervisor,
        reward_model: RewardModelSupervisor,
        batch_size: int,
        ppo_config: Optional[PPOConfig] = None,   
        initial_lr: Optional[float] = None
    ) -> None:
        super().__init__()
        self.ref_model = dialogue_model
        self.reward_model = reward_model
        self.tuned_model = copy.deepcopy(dialogue_model)
        self.ppo_config = ppo_config
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.value_head = nn.Linear(dialogue_model.model.hidden_size, 1)

    def log_probs(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
    ) -> torch.FloatTensor:
        logp = torch.gather(
            F.log_softmax(logits, dim=-1), 
            -1, 
            labels.unsqueeze(-1)).squeeze(-1)
        return logp

    @torch.no_grad
    def compute_rewards(
        self, 
        scores: torch.FloatTensor, 
        new_log_probs: torch.FloatTensor, 
        ref_log_probs: torch.FloatTensor
    ) -> torch.FloatTensor:
        kl_div = new_log_probs - ref_log_probs
        rewards = -self.ppo_config.kl_penalty * kl_div
        rewards[:, -1] += scores
        return rewards

    @torch.no_grad
    def estimate_advantages(
        self,
        rewards: torch.FloatTensor,
        values: torch.FloatTensor,
        pad_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        gen_len = torch.sum(pad_mask, dim=-1)
        advantages = torch.zeros_like(values)
        for i in range(rewards.size(dim=0)):
            len = gen_len[i]
            for j in range(len - 1, -1, -1):
                nextvalue = values[i, j + 1] if j < len - 1 else 0.0
                delta = rewards[i, j] + self.ppo_config.gamma * nextvalue - values[i, j]
                advantages[i, j] = delta + self.ppo_config.gamma * self.ppo_config._lambda * advantages[i, j + 1]
            mean, var = torch.mean(advantages[i, :len]), torch.var(advantages[i, :len])
            advantages[i, :len] = (advantages[i, :len] - mean) * torch.rsqrt(var + 1e-8)
        return advantages

    def forward_and_log_metrics(
        self,
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        stage: str
    ) -> float:

        device = batch.contexts.device if self.tuned_model.model.has_encoder else batch.dialogues.device

        with torch.no_grad():
            # ROLLOUT PHASE
            contexts = batch.raw_contexts if self.tuned_model.model.has_encoder else batch.raw_dialogues
            enc_predictions = self.tuned_model.generate(batch)
            predictions = [self.tuned_model.tokenizer.decode_to_text(enc) for enc in enc_predictions]
            
            # EVALUATION PHASE
            dialogues = [context + [prediction] for context, prediction in zip(contexts, predictions)]
            reward_model_inputs = self.reward_model.tokenizer(
                ["[CLS] " + " [SEP] ".join(dialogue) for dialogue in dialogues], 
                padding=True, 
                return_tensors="pt"
            )
            scores = self.reward_model.forward(
                RewardModelBatch(
                    dialogues=reward_model_inputs.input_ids.to(device),
                    rewards=None,
                    mask=reward_model_inputs.attention_mask.to(device)
                )
            )
        
        # OPTIMIZATION PHASE
        if self.tuned_model.model.has_encoder:
            max_len = max([len(seq) for seq in enc_predictions])
            batch.targets = torch.LongTensor(
                [seq + [self.tuned_model.tokenizer.PAD_IDX] * (max_len - len(seq)) for seq in enc_predictions])
            batch.targets = batch.targets.to(device)
        else:
            dialogues = [self.tuned_model.tokenizer.encode_text(dialogue) for dialogue in dialogues]
            max_len = max([len(seq) for seq in dialogues])
            batch.dialogues = torch.LongTensor(
                [seq + [self.tuned_model.tokenizer.PAD_IDX] * (max_len - len(seq)) for seq in dialogues])
            batch.dialogues = batch.dialogues.to(device)
        
        # Compute new and reference log probability and advantage for each generated token
        logits, last_hidden_states, target = self.tuned_model.forward(batch)
        new_log_probs = self.log_probs(logits[:, :-1, :], target)
        values = self.value_head(last_hidden_states[:, :-1, :])
        pad_mask = (target != self.tuned_model.tokenizer.PAD_IDX)
        with torch.no_grad():
            logits, _, _ = self.ref_model.forward(batch)
            ref_log_probs = self.log_probs(logits[:, :-1, :], target)

        ratios = torch.exp(new_log_probs - ref_log_probs)
        clipped_ratios = torch.clip(ratios, 1 - self.ppo_config.clip_epsilon, 1 + self.ppo_config.clip_epsilon)
        rewards = self.compute_rewards(scores, new_log_probs, ref_log_probs)
        advantages = self.estimate_advantages(rewards, values, pad_mask)
        returns = advantages + values
        
        # Apply PPO Algorithm (With policy and value function losses)
        vf_loss = (values - returns) ** 2
        clipped_values = torch.max(torch.min(
                values, 
                values + self.ppo_config.value_clip_range), 
            values - self.ppo_config.value_clip_range
        )
        vf_loss_clipped = (clipped_values - returns) ** 2
        vf_loss = torch.sum(
            torch.max(vf_loss, vf_loss_clipped) * pad_mask) / torch.sum(pad_mask)
        ppo_loss = torch.sum(
            torch.max(-ratios * advantages, -clipped_ratios * advantages) * pad_mask) / torch.sum(pad_mask)
        
        loss = ppo_loss + self.ppo_config.vf_coeff * vf_loss
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {stage: loss}, self.global_step)
        
        return loss

    def training_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss

    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {'avg_val': avg_val_loss}, self.global_step)
        return avg_val_loss

    def on_test_start(self) -> None:
        self.tuned_model.on_test_start()

    def test_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> None:
        self.tuned_model.test_step(batch)

    def test_epoch_end(self, _) -> None:
        self.tuned_model.test_epoch_end()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.8)
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

    def on_save_checkpoint(self, ckpt: OrderedDict):
        for key in list(ckpt['state_dict'].keys()):
            if key.startswith("reward_model") or key.startswith("ref_model") or key.startswith("value_head"):
                del ckpt['state_dict'][key]
            elif key.startswith("tuned_model"):
                new_key = key.replace("tuned_model.", "", 1)
                ckpt['state_dict'][new_key] = ckpt['state_dict'][key]
                del ckpt['state_dict'][key]

# -----------------------------------------------------------------------------