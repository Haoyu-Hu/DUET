"""Stand-ins for vllm.v1.sample.logits_processor types, used only when vLLM is
not importable (CPU unit tests). Field layout mirrors vLLM >= 0.10.2
(all BatchUpdate fields required, as in vLLM)."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class MoveDirectionality(enum.Enum):
    UNIDIRECTIONAL = enum.auto()
    SWAP = enum.auto()


@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int
    removed: list   # [slot]
    added: list     # [(slot, params, prompt_ids, output_ids)]
    moved: list     # [(src, dst, MoveDirectionality)]


class LogitsProcessor:
    def __init__(self, vllm_config, device, is_pin_memory):  # pragma: no cover
        raise NotImplementedError

    def apply(self, logits):  # pragma: no cover
        raise NotImplementedError

    def is_argmax_invariant(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def update_state(self, batch_update):  # pragma: no cover
        raise NotImplementedError
