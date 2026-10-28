"""Drop leading padding before the policy forward, for Qwen3.5.

SkyRL lays each micro-batch out left-padded to the longest row of the whole batch.
The Hugging Face Qwen3.5 forward gets left-padded rows wrong: on a dumped training
batch the trainer's log-probs differ from vLLM's by 0.040 per token, a plain HF
forward of the same padded row reproduces them to 6e-7, and the same row without
its padding matches vLLM to 0.006 (cs795 jobs 2987118, 2987705; worst token -9.29
vs -0.18). Every consumer of the forward reads the last ``num_actions`` columns, so
removing the leading columns that are padding in every row changes nothing else.
With ``micro_*_batch_size_per_gpu: 1`` that removes all of the padding.

Imported for its side effect by ``echo_rl.world_modeling.fsdp_worker``, which runs in
the policy worker process.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

_forward = HFModelWrapper.forward


def _unpadded_forward(self, sequences, num_actions, attention_mask=None, *args, **kwargs):
    if attention_mask is None or self.use_sample_packing:
        return _forward(self, sequences, num_actions, attention_mask, *args, **kwargs)
    # leading columns that are padding in every row
    lead = int((attention_mask.cumsum(-1) == 0).sum(-1).min())
    if lead == 0:
        return _forward(self, sequences, num_actions, attention_mask, *args, **kwargs)
    result = _forward(self, sequences[:, lead:], num_actions, attention_mask[:, lead:], *args, **kwargs)
    log_probs, output = result if isinstance(result, tuple) else (result, None)
    # Response-level tensors are num_actions wide (the longest response in the whole
    # batch). A short row trimmed of its padding yields fewer columns (2987770 failed
    # on 5548 vs 8670); the missing leading columns are padding or prompt, which the
    # loss mask excludes, so fill them with zeros.
    width = int(max(num_actions)) if not isinstance(num_actions, int) else num_actions
    if log_probs.shape[-1] < width:
        log_probs = F.pad(log_probs, (width - log_probs.shape[-1], 0))
    if output is None:
        return log_probs
    if isinstance(output.get("entropy") if hasattr(output, "get") else None, torch.Tensor):
        output["entropy"] = F.pad(output["entropy"], (lead, 0))
    return log_probs, output


if not getattr(HFModelWrapper.forward, "_echo_unpadded", False):
    _unpadded_forward._echo_unpadded = True
    HFModelWrapper.forward = _unpadded_forward
