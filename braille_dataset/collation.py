"""Custom collation functions for braille frame batches."""

from __future__ import annotations

from typing import Optional

import torch


class RandomOffsetPad:
    """Pad frame tensors to a fixed temporal length at random offsets.

    Drop-in replacement for ``tonic.collation.PadTensors`` when training with
    samples that may be shorter than the canonical recording window (e.g. after
    ``tonic.transforms.TimeSkew`` with ``coefficient<1``). Each sample of shape
    ``(T_i, *rest)`` is placed at a random offset in ``[0, n_time_bins - T_i]``
    inside a zero tensor of shape ``(n_time_bins, *rest)``, so the same stimulus
    can land in different parts of the training window across epochs.

    Parameters:
        n_time_bins (int, optional): Output length along the time axis. If
            ``None``, defaults to the longest sample in the batch (in which case
            samples shorter than the max still get a random offset within the
            slack).
        batch_first (bool): If True, output shape is ``(B, T, *rest)``; else
            ``(T, B, *rest)``.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(train_set, batch_size=32,
        ...                     collate_fn=RandomOffsetPad(n_time_bins=128))
    """

    def __init__(
        self,
        n_time_bins: Optional[int] = None,
        batch_first: bool = True,
    ):
        if n_time_bins is not None and n_time_bins <= 0:
            raise ValueError(f"n_time_bins must be positive, got {n_time_bins}")
        self.n_time_bins = n_time_bins
        self.batch_first = batch_first

    def __call__(self, batch):
        samples, targets = zip(*batch)
        tensors = [
            s if isinstance(s, torch.Tensor) else torch.as_tensor(s) for s in samples
        ]
        tail_shape = tensors[0].shape[1:]
        dtype = tensors[0].dtype
        device = tensors[0].device

        n_bins = self.n_time_bins
        if n_bins is None:
            n_bins = max(t.shape[0] for t in tensors)

        out = torch.zeros(len(batch), n_bins, *tail_shape, dtype=dtype, device=device)
        for i, t in enumerate(tensors):
            length = min(t.shape[0], n_bins)
            slack = n_bins - length
            offset = int(torch.randint(0, slack + 1, ()).item()) if slack > 0 else 0
            out[i, offset : offset + length] = t[:length]

        target_tensors = [
            y if isinstance(y, torch.Tensor) else torch.as_tensor(y) for y in targets
        ]
        if target_tensors[0].ndim > 0:
            stacked_targets = torch.stack(target_tensors, dim=0 if self.batch_first else -1)
        else:
            stacked_targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        if not self.batch_first:
            out = out.transpose(0, 1).contiguous()
        return out, stacked_targets
