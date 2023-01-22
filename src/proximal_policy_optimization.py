# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import copy
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# User-defined Modules
from dialogue_model_supervisor import DialogueModelSupervisor
from reward_model_supervisor import RewardModelSupervisor
from data_classes import DecoderModelBatch, EncoderDecoderModelBatch, RewardModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------


class PPOSupervisor(pl.LightningModule):
    def __init__(
        self,
        dialogue_model: DialogueModelSupervisor,
        reward_model: RewardModelSupervisor,
        batch_size: int,
        ppo_epsilon: Optional[float] = None,
        initial_lr: Optional[float] = None
    ) -> None:
        super().__init__()
        self.initial_model = dialogue_model
        self.reward_model = reward_model
        self.ppo_tuned_model = copy.deepcopy(dialogue_model)
        self.ppo_epsilon = ppo_epsilon
        self.initial_lr = initial_lr
        self.batch_size = batch_size

        for param in self.reward_model.model.parameters():
            param.requires_grad = False
        
        for param in self.reward_model.linear.parameters():
            param.requires_grad = False

        for param in self.initial_model.model.parameters():
            param.requires_grad = False

    def forward_and_log_metrics(
        self,
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        stage: str
    ) -> float:

        contexts = batch.raw_contexts if self.ppo_tuned_model.model.has_encoder else batch.raw_dialogues
        enc_predictions = self.ppo_tuned_model.generate(batch)
        predictions = [self.ppo_tuned_model.tokenizer.decode_to_text(enc) for enc in enc_predictions]
        dialogues = [context + [prediction] for context, prediction in zip(contexts, predictions)]
        reward_model_inputs = self.reward_model.tokenizer(
            ["[CLS] " + " [SEP] ".join(dialogue) for dialogue in dialogues], 
            padding=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            reward = self.reward_model.forward(
                RewardModelBatch(
                    dialogues=reward_model_inputs.input_ids,
                    rewards=None,
                    mask=reward_model_inputs.attention_mask
                )
            )
        if self.ppo_tuned_model.model.has_encoder:
            max_len = max([len(seq) for seq in enc_predictions])
            batch.targets = torch.LongTensor(
                [seq + [self.ppo_tuned_model.tokenizer.PAD_IDX] * (max_len - len(seq)) for seq in enc_predictions])
        else:
            dialogues = [self.ppo_tuned_model.tokenizer.encode_text(dialogue) for dialogue in dialogues]
            max_len = max([len(seq) for seq in dialogues])
            batch.dialogues = torch.LongTensor(
                [seq + [self.ppo_tuned_model.tokenizer.PAD_IDX] * (max_len - len(seq)) for seq in dialogues])
        logits, target_seq = self.ppo_tuned_model.forward(batch)
        ppo_model_log_prob = -torch.sum(F.nll_loss(
            logits[:, :-1, :].permute(0, 2, 1),
            target_seq,
            ignore_index=self.ppo_tuned_model.tokenizer.PAD_IDX,
            reduction="none"), dim=-1)
        logits, target_seq = self.initial_model.forward(batch)
        initial_model_log_prob = -torch.sum(F.nll_loss(
            logits[:, :-1, :].permute(0, 2, 1),
            target_seq,
            ignore_index=self.initial_model.tokenizer.PAD_IDX,
            reduction="none"), dim=-1)
        ratio = torch.exp(ppo_model_log_prob - initial_model_log_prob)
        loss = -torch.mean(torch.min(ratio * reward, torch.clip(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * reward))
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
        self.ppo_tuned_model.on_test_start()

    def test_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> None:
        self.ppo_tuned_model.test_step(batch)

    def test_epoch_end(self, _) -> None:
        self.ppo_tuned_model.test_epoch_end()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.8)
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

# -----------------------------------------------------------------------------