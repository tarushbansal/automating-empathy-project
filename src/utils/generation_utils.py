# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Optional, Callable, List

import torch
import torch.nn.functional as F

# User-Defined Modules
from data_classes import ModelBatch, GenerationConfig

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


def length_normalization(
    sequences: torch.LongTensor,
    stop_token: int, 
    alpha: float = 0.65
) -> float:
    N, num, len = sequences.size()
    length = torch.empty(N, num, dtype=torch.long, device=sequences.device)
    for i in range(N):
        for j in range(num):
            try:
                ind = sequences[i, j].tolist().index(stop_token)
                length[i, j] = ind + 1
            except ValueError:
                length[i, j] = len
    return length ** alpha


@torch.no_grad()
def generate(
    forward_fn: Callable,
    model_has_encoder: bool,
    batch: ModelBatch,
    stop_token: int,
    generation_config: GenerationConfig,
    start_token: Optional[int] = None,
) -> List[List[int]]:

    max_new_tokens = generation_config.max_new_tokens
    beam_width = generation_config.beam_width
    sample = generation_config.sample
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    length_alpha = generation_config.length_alpha

    N = batch.contexts.size(dim=0)
    device = batch.contexts.device

    if model_has_encoder:
        if start_token is None:
            raise ValueError("Must specify a start sequence for encoder-decoder models")
        beams = torch.LongTensor(N * [beam_width * [[start_token]]]).to(device)
        max_new_tokens += beams.size(dim=-1)
    else:
        beams = torch.empty(N, beam_width, 0, dtype=torch.long, device=device)
    
    beam_scores = torch.zeros(N, beam_width, device=device)
    beam_scores[:, 1:] = beam_scores[:, 1:].fill_(float("-inf"))
    stop_detected = torch.zeros(N, beam_width, dtype=torch.bool, device=device)
    decoder_cache = [None for _ in range(beam_width)]
    encoder_cache = None

    while True:
        # Run all beams through model and process output logits
        for i in range(beam_width):
            batch.targets = beams[:, i, :] if decoder_cache[i] is None else beams[:, i, -1:]
            input_kwargs = {
                "batch": batch,
                "encoder_outputs": encoder_cache,
                "past_key_values": decoder_cache[i],
                "use_cache": True
            }
            output, target_logits, _ = forward_fn(**input_kwargs)
            if model_has_encoder:
                encoder_cache = (output.encoder_last_hidden_state,)
            new_logits = target_logits[:, -1:, :]
            # decoder_cache[i] = output.past_key_values
            logits = torch.cat((logits, new_logits), dim=1) if i != 0 else new_logits

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
        new_scores[:, :, 0] = new_scores[:, :, 0].masked_fill(stop_detected, 0)
        new_scores[:, :, 1:] = new_scores[:, :, 1:].masked_fill(
            stop_detected.unsqueeze(-1), -float("inf"))

        # Choose {beam_width} number of sequences with highest probabilities
        new_tokens = new_tokens.view(N, beam_width ** 2).unsqueeze(-1)
        sequences = torch.cat(
            (beams.repeat_interleave(beam_width, dim=1), new_tokens), dim=-1)
        scores = (new_scores + beam_scores.unsqueeze(-1)).view(N, beam_width ** 2)
        _, top_indices = torch.topk(
            scores / length_normalization(sequences, stop_token, length_alpha), beam_width, dim=-1)
        beam_scores = torch.gather(scores, 1, top_indices)
        top_indices = top_indices.unsqueeze(-1).expand(
            N, beam_width, sequences.size(dim=-1))
        beams = torch.gather(sequences, 1, top_indices)

        # Break out of loop if stopping criteria is met
        stop_detected = torch.tensor([
            [stop_token in seq for seq in beam] for beam in beams]).to(device)
        stop = torch.all(stop_detected) or (beams.size(dim=-1) >= max_new_tokens)
        if stop:
            break

    # Return most probable sequence for each beam in batch
    sequences = beams[:, 0].tolist()
    for i in range(len(sequences)):
        try:
            ind = sequences[i].index(stop_token) + 1
        except ValueError:
            ind = len(sequences[i])
        sequences[i] = sequences[i][:ind]
    
    return sequences

# -----------------------------------------------------------------------------