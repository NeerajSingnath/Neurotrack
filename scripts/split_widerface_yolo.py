from pathlib import Path
import random, shutil

ROOT = Path(r"D:\Neurotrack\datasets\widerface_yolo")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"

assert IMAGES.exists(), f"Missing {IMAGES}"
assert LABELS.exists(), f"Missing {LABELS} (YOLO .txt labels required)"

# make split dirs
for p in [IMAGES/"train", IMAGES/"val", LABELS/"train", LABELS/"val"]:
    p.mkdir(parents=True, exist_ok=True)

# collect images
exts = {".jpg", ".jpeg", ".png"}
imgs = [p for p in IMAGES.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not imgs:
    raise SystemExit(f"No images directly under {IMAGES}. If nested, move them up first.")

# pair each image with its label
pairs, missing = [], 0
for im in imgs:
    lab = LABELS / (im.stem + ".txt")
    if lab.exists():
        pairs.append((im, lab))
    else:
        missing += 1
print(f"Found {len(pairs)} pairs; missing labels for {missing} images.")

if not pairs:
    raise SystemExit("No paired image/label files — cannot train.")

# 90/10 split
random.seed(42)
random.shuffle(pairs)
n_val = max(1, int(0.10 * len(pairs)))
val_pairs = pairs[:n_val]
train_pairs = pairs[n_val:]

def move_pair(im, lab, split):
    shutil.move(str(im), str(IMAGES/split/im.name))
    shutil.move(str(lab), str(LABELS/split/lab.name))

for im, lab in val_pairs:   move_pair(im, lab, "val")
for im, lab in train_pairs: move_pair(im, lab, "train")

print(f"Moved {len(train_pairs)} -> train/, {len(val_pairs)} -> val/. Done.")
