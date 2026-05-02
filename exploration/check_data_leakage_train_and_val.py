import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import MultiTaskBrainDataset

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_set = MultiTaskBrainDataset(
    base_dir=os.path.join(proj_dir, "brisc2025"),
    mode="classification",
    train_or_test="train",
    subset="train",
    seed=42
)

val_set = MultiTaskBrainDataset(
    base_dir=os.path.join(proj_dir, "brisc2025"),
    mode="classification",
    train_or_test="train",
    subset="val",
    seed=42
)

train_paths = set(path for path, _ in train_set.samples)
val_paths   = set(path for path, _ in val_set.samples)

overlap = train_paths & val_paths

print(f"Train samples:       {len(train_paths)}")
print(f"Val samples:         {len(val_paths)}")
print(f"Overlapping samples: {len(overlap)}")

if len(overlap) == 0:
    print("No leakage detected.")
else:
    print("LEAKAGE DETECTED:")
    for path in overlap:
        print(f"  {path}")