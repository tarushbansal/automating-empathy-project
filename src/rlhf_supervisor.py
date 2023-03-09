# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import copy
from typing import Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup

# User-defined Modules
from data_loader import pad_to_tensor
from dialogue_model_supervisor import DialogueModelSupervisor
from reward_model_supervisor import RewardModelSupervisor
from data_classes import ModelBatch, RewardModelBatch, PPOConfig

# ------------------------- IMPLEMENTATION -----------------------------------


class RLHFSupervisor(pl.LightningModule):
    def __init__(
        self,
        dialogue_model: DialogueModelSupervisor,
        reward_model: RewardModelSupervisor,
        ppo_config: Optional[PPOConfig] = None,   
        initial_lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        data_erasure_level:  Optional[float] = None,
    ) -> None:
        super().__init__()
        self.ref_model = dialogue_model
        self.reward_model = reward_model
        self.tuned_model = copy.deepcopy(dialogue_model)
        self.ppo_config = ppo_config
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.data_erasure_level = data_erasure_level
        self.dropout = nn.Dropout()
        self.value_head = nn.Linear(dialogue_model.model.hidden_size, 1)
        self.default_log_config = {
            "on_step": True,
            "on_epoch": True,
            "sync_dist": True
        }
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters({
            "config": dialogue_model.hparams["config"],
            "initial_lr": self.initial_lr,
            "batch_size": self.batch_size,
            "data_erasure_level": self.data_erasure_level,
            "generation_config": dialogue_model.generation_config.__dict__,
            "ppo_config": ppo_config.__dict__
        })

    def _log_probs(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
    ) -> torch.FloatTensor:
        logp = torch.gather(
            F.log_softmax(logits, dim=-1), 
            -1, 
            labels.unsqueeze(-1)).squeeze(-1)
        return logp

    @torch.no_grad()
    def _compute_rewards(
        self, 
        scores: torch.FloatTensor, 
        new_log_probs: torch.FloatTensor, 
        ref_log_probs: torch.FloatTensor,
        gen_len: torch.LongTensor
    ) -> torch.FloatTensor:
        kl_div = new_log_probs - ref_log_probs
        rewards = -self.ppo_config.kl_penalty * kl_div
        for i in range(gen_len.size(dim=0)):
            rewards[i, gen_len[i] - 1] += scores[i]
        return rewards

    @torch.no_grad()
    def _estimate_advantages_and_returns(
        self,
        rewards: torch.FloatTensor,
        values: torch.FloatTensor,
        gen_len: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:
        advantages = torch.zeros_like(values)
        for i in range(rewards.size(dim=0)):
            len = gen_len[i]
            nextadvantage = 0
            for j in range(len - 1, -1, -1):
                nextvalue = values[i, j + 1] if j < len - 1 else 0.0
                delta = rewards[i, j] + self.ppo_config.gamma * nextvalue - values[i, j]
                advantages[i, j] = delta + self.ppo_config.gamma * self.ppo_config.lam * nextadvantage
                nextadvantage = advantages[i, j]
        # Normalize advantage to zero mean and unit variance
        num_tokens = torch.sum(gen_len)
        mean = torch.sum(advantages) / num_tokens
        var = torch.sum(advantages ** 2) / num_tokens - mean ** 2
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8)
        
        returns = advantages + values
        
        return advantages, returns

    def _forward_and_log_metrics(
        self,
        batch: ModelBatch,
        stage: str
    ) -> float:

        N = batch.contexts.size(dim=0)
        device = batch.contexts.device

        with torch.no_grad():
            # ROLLOUT PHASE
            enc_outputs = self.tuned_model.generate(
                batch.contexts, 
                batch.context_mask, 
                default_config = True if stage == "val" else False
            )
            outputs = self.tuned_model.tokenizer.decode(enc_outputs)

            # EVALUATION PHASE
            contexts = [self.reward_model.tokenizer.encode_text(context)[0] 
                        for context in batch.raw_contexts]
            contexts, context_mask = pad_to_tensor(
                contexts,
                pad_token=self.reward_model.tokenizer.PAD_IDX
            )
            targets = [self.reward_model.tokenizer.encode_text(output)[0]
                       for output in outputs]
            targets, target_mask = pad_to_tensor(
                targets,
                pad_token=self.reward_model.tokenizer.PAD_IDX
            )
            self.reward_model.eval()
            scores = self.reward_model.forward(
                RewardModelBatch(
                    contexts=contexts.to(device),
                    context_mask=context_mask.to(device),
                    targets=targets.to(device),
                    target_mask=target_mask.to(device),
                    ratings=None
                )
            )
            self.log(f"{stage}_reward", torch.mean(scores), batch_size=N, **self.default_log_config)
        
        # OPTIMIZATION PHASE   

        # Compute new and reference log probability and advantage for each generated token
        batch.targets = enc_outputs
        batch.target_mask = (enc_outputs != self.tuned_model.tokenizer.PAD_IDX)
        output, _ = self.tuned_model.forward(batch)
        target_logits = (
            output.logits[:, :-1, :] 
            if self.tuned_model.model.has_encoder 
            else output.logits[:, batch.contexts.size(dim=1)-1:-1]
        )
        last_hidden_states = (
            output.decoder_hidden_states[-1][:, :-1, :]
            if self.tuned_model.model.has_encoder
            else output.hidden_states[-1][:, batch.contexts.size(dim=1)-1:-1]
        )
        labels = batch.targets[:, 1:] if self.tuned_model.model.has_encoder else batch.targets
        mask = batch.target_mask[:, 1:] if self.tuned_model.model.has_encoder else batch.target_mask
        gen_len = torch.sum(mask, dim=-1)
        new_log_probs = self._log_probs(target_logits, labels)
        entropy = -torch.sum(torch.softmax(target_logits, dim=-1) * 
                             torch.log_softmax(target_logits, dim=-1), dim=-1)
        values = self.value_head(self.dropout(last_hidden_states)).squeeze(-1)

        with torch.no_grad():
            self.ref_model.eval()
            output, _ = self.ref_model.forward(batch)
            target_logits = (
                output.logits[:, :-1, :] 
                if self.tuned_model.model.has_encoder 
                else output.logits[:, batch.contexts.size(dim=1)-1:-1]
            )
            ref_log_probs = self._log_probs(target_logits, labels)

        ratios = torch.exp(new_log_probs - ref_log_probs)
        clipped_ratios = torch.clip(
            ratios, 1 - self.ppo_config.clip_epsilon, 1 + self.ppo_config.clip_epsilon)
        rewards = self._compute_rewards(scores, new_log_probs, ref_log_probs, gen_len)
        advantages, returns = self._estimate_advantages_and_returns(rewards, values, gen_len)

        # Apply PPO Algorithm (With ppo clip, value function and entropy losses)
        num_tokens = torch.sum(mask)
        vf_loss = torch.sum(((values - returns) ** 2) * mask) / num_tokens
        entropy_loss = -torch.sum(entropy * mask) / num_tokens
        ppo_loss = torch.sum(
            torch.max(-ratios * advantages, 
                      -clipped_ratios * advantages) * mask
            ) / num_tokens
        loss = (
            ppo_loss + 
            self.ppo_config.vf_coeff * vf_loss + 
            self.ppo_config.entropy_coeff * entropy_loss
        )
        
        self.log(f"{stage}_ppo_loss", ppo_loss, batch_size=N, **self.default_log_config)
        self.log(f"{stage}_vf_loss", vf_loss, batch_size=N, **self.default_log_config)
        self.log(f"{stage}_entropy_loss", entropy_loss, batch_size=N, **self.default_log_config)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=N, **self.default_log_config)
        
        return loss

    def training_step(
        self, 
        batch: ModelBatch,
        _
    ) -> float:
        train_loss = self._forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(
        self, 
        batch: ModelBatch,
        _
    ) -> float:
        val_loss = self._forward_and_log_metrics(batch, "val")
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.initial_lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.max_epochs
        )
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

    def on_save_checkpoint(self, ckpt: OrderedDict):
        for key in list(ckpt['state_dict'].keys()):
            if (key.startswith("reward_model") 
                or key.startswith("ref_model") 
                or key.startswith("value_head")):
                del ckpt['state_dict'][key]
            elif key.startswith("tuned_model"):
                new_key = key.replace("tuned_model.", "", 1)
                ckpt['state_dict'][new_key] = ckpt['state_dict'][key]
                del ckpt['state_dict'][key]

# -----------------------------------------------------------------------------