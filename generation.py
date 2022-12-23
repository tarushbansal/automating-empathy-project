# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Union, Optional, Callable

import torch
import torch.nn.functional as F

# User-Defined Modules
from data_classes import EncoderDecoderModelBatch, DecoderModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------

def warp_logits(
    logits: torch.LongTensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50
) -> torch.LongTensor:

    vocab_size = logits.size(dim=-1)
    if temperature != 1.0:
        logits /= temperature
    if top_p != 1.0:
        sorted_logits, sorted_ind = torch.sort(logits, dim=-1, descending=False)
        cummulative_logits = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted_logits = (cummulative_logits < (1 - top_p))
        remove_logits = remove_sorted_logits.scatter(-1, sorted_ind, remove_sorted_logits)
        logits = logits.masked_fill(remove_logits, float("-inf"))
    if top_k < vocab_size:
        remove_logits = logits < torch.topk(logits, top_k, dim=-1)[0][:, :, -1:]
        logits = logits.masked_fill(remove_logits, float("-inf"))

    return logits
    

@torch.no_grad()
def generate(
    forward_fn: Callable,
    model_has_encoder: bool,
    batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
    stop_token: int,
    pad_token: int,
    vocab_size: int,
    max_new_tokens: int = 100,
    beam_width: int = 1,
    sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    start_token: Optional[int] = None,
) -> torch.LongTensor:

    if model_has_encoder:
        N = batch.contexts.size(dim=0)
        device = batch.contexts.device
    else:
        N = batch.dialogues.size(dim=0)
        dialogues = batch.dialogues
        device = batch.dialogues.device
        last_non_masked_idx = torch.sum(batch.dialogues != pad_token, dim=1) - 1
        start_idx = last_non_masked_idx[:, None, None].expand(-1, -1, vocab_size).clone()

    if start_token is None:
        beams = torch.empty(N, beam_width, 0, dtype=torch.long, device=device)
    else:
        beams = torch.empty(
            N, beam_width, 1, 
            dtype=torch.long, device=device).fill_(start_token)
        max_new_tokens += 1
    
    beam_scores = torch.zeros(N, beam_width, device=device)
    beam_scores[:, 1:] = beam_scores[:, 1:].fill_(float("-inf"))
    stop_detected = torch.zeros(N, beam_width, dtype=torch.bool, device=device)

    while True:
        # Run all beams through model and process output logits
        for i in range(beam_width):
            if model_has_encoder:
                batch.targets = beams[:, i, :]
                new_logits = forward_fn(batch)[:, -1, :].unsqueeze(1)
            else:
                unstacked_sequences = tuple(torch.cat((
                    dialogues[j, :last_non_masked_idx[j]+1], 
                    beams[j, i, :], 
                    dialogues[j, last_non_masked_idx[j]+1:]), dim=-1) for j in range(N))
                batch.dialogues = torch.stack(unstacked_sequences)
                new_logits = forward_fn(batch).gather(1, start_idx)
            logits = torch.cat((logits, new_logits), dim=1) if i != 0 else new_logits
        
        if not model_has_encoder:
            start_idx += 1
            
        warped_logits = warp_logits(logits, temperature, top_p, top_k)
        if sample:
            new_tokens = torch.multinomial(
                F.softmax(warped_logits.view(N * beam_width, -1), dim=-1),
                beam_width,
            )
            new_tokens = new_tokens.view(N, beam_width, beam_width)
            new_scores = F.log_softmax(warped_logits, dim=-1).gather(-1, new_tokens)
        else:
            new_scores, new_tokens = torch.topk(
                F.log_softmax(warped_logits, dim=-1),
                beam_width,
                dim=-1
            )
        new_scores = new_scores.masked_fill(stop_detected.unsqueeze(-1), 0)

        # Choose {beam_width} number of sequences with highest probabilities
        new_tokens = new_tokens.view(N, beam_width ** 2).unsqueeze(-1)
        sequences = torch.cat(
            (beams.repeat_interleave(beam_width, dim=1), new_tokens), dim=-1)
        scores = (new_scores + beam_scores.unsqueeze(-1)).view(N, beam_width ** 2)
        beam_scores, top_indices = torch.topk(scores, beam_width, dim=-1)
        top_indices = top_indices.unsqueeze(-1).expand(
            N, beam_width, sequences.size(dim=-1))
        beams = torch.gather(sequences, 1, top_indices)

        # Break out of loop if stopping criteria is met
        stop_detected = torch.tensor([
            [stop_token in seq for seq in beam] for beam in beams]).to(device)
        stop = torch.all(stop_detected) or (beams.size(dim=-1) >= max_new_tokens)
        if stop:
            break

    # Return most probable sequence and its log probability for each beam in batch
    return beams[:, 0]

# -----------------------------------------------------------------------------