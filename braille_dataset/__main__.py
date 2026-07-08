import tonic.transforms as transforms
from torch.utils.data import DataLoader

from braille_dataset import Braille, RandomOffsetPad

if __name__ == "__main__":
    full = Braille(save_to="./data", threshold=2)
    print(f"full: {len(full)} samples, {len(full.classes)} classes, "
          f"sensor_size={full.sensor_size}")

    # Cross-validation fold 1 + fixed test set
    train1 = Braille(save_to="./data", threshold=2, split="train1")
    val1 = Braille(save_to="./data", threshold=2, split="val1")
    test = Braille(save_to="./data", threshold=2, split="test")
    print(f"train1={len(train1)}  val1={len(val1)}  test={len(test)}")

    events, label = train1[0]
    print(f"raw events: shape={events.shape}, label={label} ({train1.decode_label(label)})")

    # Tonic frame pipeline
    pipeline = transforms.Compose([
        transforms.TimeJitter(std=5000, clip_negative=True),
        transforms.ToFrame(sensor_size=train1.sensor_size, n_time_bins=128),
    ])
    train1.transform = pipeline
    loader = DataLoader(
        train1, batch_size=8, shuffle=True,
        collate_fn=RandomOffsetPad(n_time_bins=128),
    )
    batch, targets = next(iter(loader))
    print(f"batch shape (B, T, P, W): {tuple(batch.shape)}, targets: {targets.shape}")
