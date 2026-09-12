"""Classes de Dataset e DataLoader customizadas (janelas deslizantes).

BdtdDataset: dado um corpus ja tokenizado (lista de ids), retorna pares
input-target usando janelas deslizantes com max_length e stride.

BdtdDataLoader: monta os registros em batches; opcionalmente embaralha
os indices a cada epoch.
"""

from __future__ import annotations

import math
import random

import torch
from torch.utils.data import Dataset


class BdtdDataset(Dataset):
    def __init__(self, tokens: list[int], max_length: int = 6, stride: int = 1):
        if max_length < 1 or stride < 1:
            raise ValueError("max_length e stride devem ser >= 1")
        if len(tokens) <= max_length:
            raise ValueError("corpus precisa de mais tokens que max_length")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        self._starts = range(0, len(self.tokens) - max_length, stride)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._starts[idx]
        end = start + self.max_length
        inputs = self.tokens[start:end]
        targets = self.tokens[start + 1 : end + 1]
        return inputs, targets


class BdtdDataLoader:
    def __init__(
        self, dataset: BdtdDataset, batch_size: int = 2, shuffle: bool = True, seed: int | None = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            inputs = torch.stack([self.dataset[i][0] for i in batch])
            targets = torch.stack([self.dataset[i][1] for i in batch])
            yield inputs, targets


def describe_batch(inputs: torch.Tensor, targets: torch.Tensor) -> str:
    return (
        f"inputs shape={tuple(inputs.shape)} dtype={inputs.dtype}, "
        f"targets shape={tuple(targets.shape)} dtype={targets.dtype}"
    )