# Spiking Braille Dataset

`spiking-braille-dataset` is a small PyTorch/tonic wrapper for the event-based tactile Braille dataset introduced in [Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern Recognition on Neuromorphic Hardware](https://arxiv.org/pdf/2205.15864) by Muller-Cleve et al. (2022). The original dataset is available on Zenodo with DOI [10.5281/zenodo.7050094](https://zenodo.org/records/7050094).

The dataset was recorded by sliding a sensorized iCub fingertip over 3D-printed Braille letters. This repository exposes the recordings as a [tonic](https://tonic.readthedocs.io/)-compatible `Dataset` class, adds reproducible train/validation/test splits, and includes small helpers for common frame-based training pipelines.

## Dataset Overview

- **Samples:** 5,400 total, with 200 repetitions per class
- **Classes:** 27 labels: letters `A` to `Z` plus `Space`
- **Sensor:** 12 capacitive taxels on the iCub fingertip
- **Polarities:** 2 event polarities, `OFF` and `ON`
- **Recording window:** 51 frames at 40 Hz, or 1.275 s
- **Motion:** 15.5 mm sliding distance at 20 mm/s
- **Encoding thresholds:** 1, 2, 5, and 10

The original release provides four sigma-delta encoding thresholds. Higher thresholds produce sparser event streams, which increases compression but also increases reconstruction error.

| Threshold | Mean events/sample | Compression ratio | Reconstruction MSE |
|-----------|--------------------|-------------------|--------------------|
| 1         | 87.6               | 1.0               | 0.0                |
| 2         | 38.0               | 2.3               | 0.4                |
| 5         | 10.5               | 8.3               | 3.7                |
| 10        | 3.4                | 25.7              | 12.3               |

## Setup

Download the repository ZIP archive from <https://anonymous.4open.science/r/spiking-braille/>, then install it locally:

```bash
unzip spiking-braille.zip
cd spiking-braille
pip install -e .
```

The dataset is downloaded automatically from Zenodo the first time you instantiate `Braille`:

```python
from braille_dataset import Braille

dataset = Braille(save_to="./data", threshold=2)
print(len(dataset))
```

This creates `./data/Braille/` and stores the threshold-specific pickle files there. To populate the cache manually, download `reading_braille_data.zip` from [Zenodo](https://zenodo.org/records/7050094), extract it into `data/Braille/`, and make sure files such as `data_braille_letters_th_1.pkl` are directly inside that folder.

You can also run the package smoke test:

```bash
python -m braille_dataset
```

## Usage
```python
import tonic.transforms as transforms
from braille_dataset import Braille

dataset = Braille(save_to="./data", threshold=2)
events, label = dataset[0]

print(events.dtype.names, events.shape, label, dataset.decode_label(label))
# ('t', 'x', 'p') (592,) 0 A

# Frames in tonic's (T, P, W) layout for 1D sensors.
dataset.transform = transforms.ToFrame(
    sensor_size=dataset.sensor_size,
    n_time_bins=128,
)
frames, label = dataset[0]
print(frames.shape)
# (128, 2, 12)
```

## Train, Validation, and Test Splits

Pass `split=` to select a partition:

| Value                 | Size        | Description                                      |
|-----------------------|-------------|--------------------------------------------------|
| `"train1"`-`"train5"` | 3,780 (70%) | Training set for the K-th cross-validation fold |
| `"val1"`-`"val5"`     | 810 (15%)   | Validation set paired with `trainK`              |
| `"test"`              | 810 (15%)   | Fixed hold-out shared across all 5 folds         |
| `None`                | 5,400       | Full dataset                                     |

`valK` and `test` are stratified with exactly 30 samples per class each. The five validation folds are pairwise disjoint and are also disjoint from `test`. For each fold, `trainK = all - test - valK`.

Split indices are stored in `braille_dataset/splits.json`. The file was generated once with `scripts/generate_splits.py` so the published splits do not depend on RNG behavior.

## DataLoader Example

The original augmentations map directly onto built-in tonic transforms:

| Original augmentation       | Tonic equivalent                                  |
|-----------------------------|---------------------------------------------------|
| `drop_prob=0.1`             | `transforms.DropEvent(p=0.1)`                     |
| `jitter_sigma_s=0.005`      | `transforms.TimeJitter(std=5000)` in microseconds |
| `contract_range=(0.9, 1.0)` | `transforms.TimeSkew(coefficient=(0.9, 1.0))`     |

Putting the splits, augmentations, framing, and random-offset frame collator together:

```python
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from braille_dataset import Braille, RandomOffsetPad

train_pipeline = transforms.Compose([
    transforms.DropEvent(p=0.1),
    transforms.TimeJitter(std=5000, clip_negative=True),
    transforms.TimeSkew(coefficient=(0.9, 1.0)),
    transforms.ToFrame(sensor_size=Braille.sensor_size, n_time_bins=128),
])
eval_pipeline = transforms.ToFrame(sensor_size=Braille.sensor_size, n_time_bins=128)

train = Braille(save_to="./data", threshold=2, split="train1", transform=train_pipeline)
val = Braille(save_to="./data", threshold=2, split="val1", transform=eval_pipeline)
test = Braille(save_to="./data", threshold=2, split="test", transform=eval_pipeline)

train_loader = DataLoader(
    train,
    batch_size=128,
    shuffle=True,
    collate_fn=RandomOffsetPad(n_time_bins=128),
)
val_loader = DataLoader(
    val,
    batch_size=128,
    collate_fn=RandomOffsetPad(n_time_bins=128),
)
test_loader = DataLoader(
    test,
    batch_size=128,
    collate_fn=RandomOffsetPad(n_time_bins=128),
)
```

`RandomOffsetPad` is useful after temporal contraction. It pads each frame tensor at a random temporal offset instead of always left-aligning it, so the same stimulus can appear in different parts of the training window across epochs.

## API Reference

### `Braille`

```python
Braille(
    save_to,
    threshold=1,
    split=None,
    transform=None,
    target_transform=None,
    transforms=None,
)
```

- `save_to`: download/cache directory. Files are stored in `save_to/Braille/`.
- `threshold`: sigma-delta threshold, one of `{1, 2, 5, 10}`.
- `split`: one of `"train1"` to `"train5"`, `"val1"` to `"val5"`, `"test"`, or `None`.
- `transform`: callable applied to the event array.
- `target_transform`: callable applied to the integer target.
- `transforms`: callable applied jointly to `(events, target)`.

Helpers and class attributes:

- `dataset.decode_label(idx) -> str`: map an integer label back to its class name.
- `Braille.sensor_size == (12, 1, 2)`: tonic sensor shape `(W, H, P)`.
- `Braille.classes`: list of 27 label strings.
- `Braille.recording_duration_us`: native recording duration in microseconds.

### Sample Format

Each call to `dataset[i]` returns `(events, target)`, where `events` is a structured NumPy array sorted by timestamp:

| Field | Meaning                           |
|-------|-----------------------------------|
| `t`   | Timestamp in microseconds         |
| `x`   | Taxel index in `[0, 12)`          |
| `p`   | Polarity: `0 = OFF`, `1 = ON`     |

With `tonic.transforms.ToFrame(sensor_size=(12, 1, 2), n_time_bins=N)`, the result has shape `(N, 2, 12)`, corresponding to `(time, polarity, taxel)`. This follows tonic's convention for 1D sensors: the singleton `H` axis is dropped, as in datasets such as `SHD` and `NTIDIGITS18` where events have no `y` field.

To collapse both polarities into one channel, use `tonic.transforms.MergePolarities()` before `ToFrame`, yielding `(N, 1, 12)`.

### Additional Helpers

- `RandomOffsetPad(n_time_bins=None, batch_first=True)`: collate framed samples into a batch and place shorter samples at random temporal offsets inside the output window.
- `RepeatSequence(n_repeats, duration_us=Braille.recording_duration_us)`: repeat an event sequence along the time axis by shifting timestamps for each copy.

## Citation

If you use this wrapper, please cite the original dataset and paper:

- Muller-Cleve et al. (2022), [Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern Recognition on Neuromorphic Hardware](https://arxiv.org/pdf/2205.15864)
- Original dataset: [10.5281/zenodo.7050094](https://zenodo.org/records/7050094)
