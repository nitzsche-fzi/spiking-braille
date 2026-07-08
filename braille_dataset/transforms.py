"""Braille-specific event transforms that follow tonic conventions.

The original braille augmentations (random event drop, Gaussian temporal
jitter, uniform temporal contraction) all map cleanly onto built-in tonic
transforms - use ``tonic.transforms.DropEvent``, ``tonic.transforms.TimeJitter``
(``std`` in microseconds), and ``tonic.transforms.TimeSkew`` with a
``coefficient`` range respectively. This module only adds the pieces that
have no tonic equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Braille


@dataclass(frozen=True)
class RepeatSequence:
    """Tile the spike sequence ``n_repeats`` times along the time axis.

    Each repeat is a copy of the events with timestamps shifted by
    ``duration_us`` (microseconds). Useful to give a network multiple passes
    over the same stimulus before classification.

    Parameters:
        n_repeats (int): Number of total copies (1 = no-op).
        duration_us (int): Length of one repeat in microseconds. Defaults to
            the canonical braille recording duration (1.275 s).
    """

    n_repeats: int
    duration_us: int = Braille.recording_duration_us

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if self.n_repeats <= 1:
            return events
        if events.size == 0:
            return events
        copies = [events.copy()]
        for i in range(1, self.n_repeats):
            c = events.copy()
            c["t"] = c["t"] + i * int(self.duration_us)
            copies.append(c)
        return np.concatenate(copies)
