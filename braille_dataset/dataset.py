from __future__ import annotations

import json
import os
import pickle
import zipfile
from typing import Callable, List, Optional

import numpy as np
import requests

from tonic.dataset import Dataset
from tonic.io import make_structured_array


_DATA_URL = "https://zenodo.org/records/7050094/files/reading_braille_data.zip?download=1"
_N_TAXELS = 12
_RECORDING_DURATION_S = 1.275
_RECORDING_DURATION_US = int(_RECORDING_DURATION_S * 1e6)
_SPLITS_PATH = os.path.join(os.path.dirname(__file__), "splits.json")


class Braille(Dataset):
    """Reading-Braille tactile event dataset, in tonic-compatible format.

    Sliding-finger recordings of 27 Braille letters (A-Z + space) captured by 12
    capacitive taxels on an iCub fingertip. The raw signal is sigma-delta
    encoded into ON/OFF spikes at four thresholds.

    ::

        @inproceedings{muller2022braille,
          title={Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern
                 Recognition on Neuromorphic Hardware},
          author={Muller-Cleve, Simon F and others},
          year={2022}
        }

    Each sample is a tuple ``(events, target)`` where ``events`` is a structured
    numpy array with fields ``t`` (microseconds), ``x`` (taxel index in
    ``[0, 12)``), ``p`` (polarity: 0=OFF, 1=ON), sorted by ``t``. ``target`` is
    an integer class label.

    Parameters:
        save_to (str): Location to save files to on disk. The dataset lives in
            a ``Braille/`` subfolder.
        threshold (int): Sigma-delta encoding threshold. One of {1, 2, 5, 10}.
            Higher thresholds give sparser event streams.
        split (str, optional): One of ``"train1".."train5"``, ``"val1".."val5"``,
            ``"test"``, or ``None`` (the full dataset). ``trainK`` / ``valK`` are
            the train/val partitions of the K-th of 5 cross-validation folds
            (70% / 15% per fold, paired together). ``test`` is a single fixed
            15% hold-out shared across all folds. Val and test are stratified
            (30 samples per class each). Indices come from a JSON file shipped
            with the package, so they are reproducible without depending on
            RNG-seed consistency. Regenerate with ``scripts/generate_splits.py``.
        transform (callable, optional): A callable applied to events.
        target_transform (callable, optional): A callable applied to targets.
        transforms (callable, optional): A callable applied to (events, target).
    """

    base_url = "https://zenodo.org/records/7050094/files/"
    filename = "reading_braille_data.zip"
    folder_name = ""

    THRESHOLDS = (1, 2, 5, 10)
    classes = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["Space"]

    # 12 taxels (x), 1 row (y), 2 polarities (ON/OFF)
    sensor_size = (_N_TAXELS, 1, 2)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    # Native recording duration in microseconds (constant per the dataset spec).
    recording_duration_us = _RECORDING_DURATION_US

    SPLITS = (
        "train1", "train2", "train3", "train4", "train5",
        "val1",   "val2",   "val3",   "val4",   "val5",
        "test",
    )

    def __init__(
        self,
        save_to: str,
        threshold: int = 1,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        if threshold not in self.THRESHOLDS:
            raise ValueError(
                f"threshold must be one of {self.THRESHOLDS}, got {threshold}"
            )
        if split is not None and split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS} or None, got {split!r}")
        self.threshold = threshold
        self.split = split
        self.url = _DATA_URL

        if not self._check_exists():
            self.download()

        pkl_path = os.path.join(
            self.location_on_system, f"data_braille_letters_th_{threshold}.pkl"
        )
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        raw_events = raw["events"].tolist()
        raw_labels = raw["letter"].tolist()

        unique_labels = sorted(set(raw_labels))
        self._label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self._idx_to_label = unique_labels

        self.data: List[np.ndarray] = [self._to_structured(ev) for ev in raw_events]
        self.targets: List[int] = [self._label_to_idx[lbl] for lbl in raw_labels]

        if split is None:
            self._indices: np.ndarray = np.arange(len(self.targets), dtype=np.int64)
        else:
            self._indices = self._load_split_indices(split, expected_n=len(self.targets))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @classmethod
    def _to_structured(cls, event_list) -> np.ndarray:
        """Flatten nested ``[taxel][polarity] -> [times_s]`` to a structured array.

        Times are converted seconds -> microseconds (int64) and the result is
        sorted by ``t`` so it matches tonic's convention for event arrays.
        """
        ts, xs, ps = [], [], []
        for x, taxel in enumerate(event_list):
            for p, group in enumerate(taxel):
                arr = np.asarray(group, dtype=np.float64)
                if arr.size == 0:
                    continue
                ts.append(np.round(arr * 1e6).astype(np.int64))
                xs.append(np.full(arr.shape, x, dtype=np.int64))
                ps.append(np.full(arr.shape, p, dtype=np.int64))
        if not ts:
            return np.empty(0, dtype=cls.dtype)
        t = np.concatenate(ts)
        x = np.concatenate(xs)
        p = np.concatenate(ps)
        order = np.argsort(t, kind="stable")
        return make_structured_array(t[order], x[order], p[order], dtype=cls.dtype)

    def decode_label(self, idx: int) -> str:
        """Map an integer label back to its letter/digit string."""
        return self._idx_to_label[idx]

    # ------------------------------------------------------------------
    # tonic.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        idx = int(self._indices[index])
        events = self.data[idx].copy()
        target = self.targets[idx]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def _check_exists(self) -> bool:
        pkl_path = os.path.join(
            self.location_on_system,
            f"data_braille_letters_th_{self.threshold}.pkl",
        )
        return os.path.isfile(pkl_path)

    def download(self) -> None:
        os.makedirs(self.location_on_system, exist_ok=True)
        zip_path = os.path.join(self.location_on_system, self.filename)
        print(f"Downloading {self.filename} from {self.url} ...")
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.location_on_system)
        os.remove(zip_path)

    # ------------------------------------------------------------------
    # Split loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_split_indices(split: str, expected_n: int) -> np.ndarray:
        if not os.path.isfile(_SPLITS_PATH):
            raise FileNotFoundError(
                f"Split file {_SPLITS_PATH} missing. Regenerate with "
                "scripts/generate_splits.py."
            )
        with open(_SPLITS_PATH) as f:
            payload = json.load(f)
        if payload.get("n_samples") != expected_n:
            raise RuntimeError(
                f"splits.json was generated for n_samples={payload.get('n_samples')}, "
                f"but the loaded dataset has {expected_n} samples. Regenerate "
                "with scripts/generate_splits.py."
            )
        if split not in payload["splits"]:
            raise KeyError(f"split {split!r} not present in splits.json")
        return np.asarray(payload["splits"][split], dtype=np.int64)
