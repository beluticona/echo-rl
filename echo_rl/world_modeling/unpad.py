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
    if isinstance(result, tuple):
        log_probs, output = result
        if isinstance(output.get("entropy") if hasattr(output, "get") else None, torch.Tensor):
            output["entropy"] = F.pad(output["entropy"], (lead, 0))
        return log_probs, output
    return result


if not getattr(HFModelWrapper.forward, "_echo_unpadded", False):
    _unpadded_forward._echo_unpadded = True
    HFModelWrapper.forward = _unpadded_forward
