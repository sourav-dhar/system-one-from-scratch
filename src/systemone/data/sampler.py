"""Shuffled mini-batches from an in-memory :class:`PairSet`, prefetched on a background thread.

Batches are gathered from host memory and copied to the device while the previous step runs on
the GPU, which keeps a single L40S busy at batch 16,384 without a separate data-loader process.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

from systemone.data.pairs import PairSet


class BatchSampler:
    """Infinite iterator of device batches. Each epoch is a fresh permutation; ragged tails are dropped."""

    def __init__(
        self,
        data: PairSet,
        batch_size: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
        prefetch: int = 2,
    ) -> None:
        if batch_size > len(data):
            raise ValueError(f"batch_size {batch_size} exceeds dataset size {len(data)}")
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.prefetch = prefetch
        self.epoch = 0
        self._tensors = {
            "states": torch.from_numpy(np.asarray(data.states)),
            "actions": torch.from_numpy(np.asarray(data.actions)),
            "doc": torch.from_numpy(np.asarray(data.doc)),
        }
        if data.negatives is not None and data.neg_count is not None:
            self._tensors["negatives"] = torch.from_numpy(np.asarray(data.negatives))
            self._tensors["neg_count"] = torch.from_numpy(np.asarray(data.neg_count))

    @property
    def steps_per_epoch(self) -> int:
        return len(self.data) // self.batch_size

    def _indices(self) -> Iterator[tuple[int, Tensor]]:
        while True:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.data), generator=g)
            for j in range(self.steps_per_epoch):
                idx, _ = perm[j * self.batch_size : (j + 1) * self.batch_size].sort()
                yield self.epoch, idx
            self.epoch += 1

    def _gather(self, idx: Tensor) -> dict[str, Tensor]:
        non_blocking = self.device.type == "cuda"
        return {
            k: v.index_select(0, idx).to(self.device, non_blocking=non_blocking)
            for k, v in self._tensors.items()
        }

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        if self.prefetch <= 0:
            for _, idx in self._indices():
                yield self._gather(idx)
            return
        q: queue.Queue[dict[str, Tensor]] = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def work() -> None:
            for _, idx in self._indices():
                batch = self._gather(idx)
                while not stop.is_set():
                    try:
                        q.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return

        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                yield q.get()
        finally:
            stop.set()
