# D:\Neurotrack\scripts\train_yolo_faces.py
from ultralytics import YOLO
from pathlib import Path
import yaml, torch

def main():
    ROOT = Path(r"D:\Neurotrack")
    DATASET = ROOT / "datasets" / "widerface_yolo"
    DATA_YAML = ROOT / "datasets" / "widerface_yolo.yaml"
    MODELS_DIR = ROOT / "models"
    RUNS_DIR = ROOT / "runs"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        with open(DATA_YAML, "w") as f:
            yaml.safe_dump({
                "path": "D:/Neurotrack/datasets/widerface_yolo",
                "train": "images/train",
                "val": "images/val",
                "names": ["face"]
            }, f)

    device_arg = 0 if torch.cuda.is_available() else "cpu"

    #model = YOLO("yolov8n.pt")
    # model = YOLO("yolov8n.pt")   # <-- old, starts from scratch
    model = YOLO(str(RUNS_DIR / "yolo_faces" / "weights" / "last.pt"))
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device_arg,
        workers=2,                    # 👈 keep 2 if you use the main-guard
        project=str(RUNS_DIR.as_posix()),
        name="yolo_faces",
        resume=True 
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = MODELS_DIR / "yolo_faces_best.pt"
    if best.exists():
        out.write_bytes(best.read_bytes())
        print(f"[OK] Saved detector weights -> {out}")
    else:
        print("[WARN] best.pt not found; check:", results.save_dir)

if __name__ == "__main__":     # 👈 Windows-safe guard
    main()
