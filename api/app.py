"""# api/app.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import tempfile, shutil, io, json, cv2, torch, numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO
import math
import time

# ---------- Paths & Globals ----------
# If you prefer robust relative paths, comment next line and use ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(r"D:\Neurotrack")
# ROOT = Path(__file__).resolve().parents[1]  # alternative: auto-detect project root
MODELS = ROOT / "models"

app = FastAPI(title="NeuroTrack Inference API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- Load Models ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO face detector
yolo = YOLO(str(MODELS / "yolo_faces_best.pt"))

# Emotion labels
with open(MODELS / "emotion_labels.json", "r", encoding="utf-8") as f:
    EMO_LABELS: List[str] = json.load(f)

# Emotion classifier (MobileNetV2 head)
emo_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import tempfile, shutil, io, json, cv2, torch, numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO
import math
import time

# ---------- Paths & Globals ----------

# Detect environment: local vs container (Render)
# Render (Linux) runs from /app, local usually from D:\ or C:\ paths
try:
    ROOT = Path(__file__).resolve().parents[1]
except Exception:
    ROOT = Path.cwd()

# If models folder exists locally, use that; else fallback to /app/models
MODELS = ROOT / "models"
if not MODELS.exists():
    MODELS = Path("/app/models")

# ---------- FastAPI init ----------
app = FastAPI(title="NeuroTrack Inference API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Device selection ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {DEVICE}")
print(f"🔹 Models path: {MODELS}")

# ---------- Load Models ----------

# YOLO face detector
yolo_path = MODELS / "yolo_faces_best.pt"
if not yolo_path.exists():
    raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
yolo = YOLO(str(yolo_path))

# Emotion labels
emo_labels_path = MODELS / "emotion_labels.json"
if not emo_labels_path.exists():
    raise FileNotFoundError(f"Emotion labels file not found at: {emo_labels_path}")

with open(emo_labels_path, "r", encoding="utf-8") as f:
    EMO_LABELS: List[str] = json.load(f)

# Emotion classifier (MobileNetV2 head)
emo_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
in_feats = emo_model.classifier[1].in_features
emo_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_feats, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, len(EMO_LABELS)),
)
emo_state_path = MODELS / "emotion_mbv2_best.pt"
if not emo_state_path.exists():
    raise FileNotFoundError(f"Emotion model weights not found at: {emo_state_path}")

state = torch.load(emo_state_path, map_location=DEVICE)
emo_model.load_state_dict(state)
emo_model.eval().to(DEVICE)


in_feats = emo_model.classifier[1].in_features
emo_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_feats, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, len(EMO_LABELS)),
)
state = torch.load(MODELS / "emotion_mbv2_best.pt", map_location=DEVICE)
emo_model.load_state_dict(state)
emo_model.eval().to(DEVICE)

emo_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

# ---------- Heuristics / Mappings ----------
# Simple stress weighting (tune as needed)
EMO_WEIGHTS: Dict[str, float] = {
    "angry":   1.0,
    "fear":    1.0,
    "sad":     1.0,
    "disgust": 0.4,
    "neutral": 0.3,
    "surprise":0.2,
    "happy":  -0.2,  # happiness reduces stress score slightly
}
STRESS_THRESHOLD = 0.50

# Rough circumplex (valence, arousal) mapping per emotion label ([-1,1] scale)
VAL_AROUSAL: Dict[str, Tuple[float, float]] = {
    "angry":   (-0.8,  0.8),
    "disgust": (-0.7,  0.3),
    "fear":    (-0.9,  0.9),
    "happy":   ( 0.8,  0.6),
    "neutral": ( 0.0,  0.0),
    "sad":     (-0.8, -0.4),
    "surprise":( 0.2,  0.9),
}

# ---------- Helpers ----------
def shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy (base e) for a probability vector."""
    p = np.array(probs, dtype=np.float32)
    p = p[(p > 0) & (p <= 1)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())

def topk_from_probs(labels: List[str], probs: List[float], k: int = 3) -> List[Dict[str, float]]:
    pairs = [{"label": labels[i], "prob": float(p)} for i, p in enumerate(probs)]
    pairs.sort(key=lambda d: d["prob"], reverse=True)
    return pairs[:max(1, k)]

def stress_score_from_probs(labels: List[str], probs: List[float]) -> float:
    score = 0.0
    for i, p in enumerate(probs):
        w = EMO_WEIGHTS.get(labels[i], 0.0)
        score += w * p
    # clamp to [0,1] for interpretability
    return float(max(0.0, min(1.0, score)))

def valence_arousal_from_probs(labels: List[str], probs: List[float]) -> Tuple[float, float]:
    v, a = 0.0, 0.0
    for i, p in enumerate(probs):
        vv, aa = VAL_AROUSAL.get(labels[i], (0.0, 0.0))
        v += p * vv
        a += p * aa
    # already bounded-ish by construction
    return float(v), float(a)

def moving_average(arr: List[List[float]], k: int) -> List[List[float]]:
    """Simple causal moving average over a list of vectors (len(labels))."""
    if k <= 1 or len(arr) == 0:
        return arr
    out = []
    cumsum = np.zeros(len(arr[0]), dtype=np.float32)
    q: List[np.ndarray] = []
    for vec in arr:
        v = np.array(vec, dtype=np.float32)
        q.append(v)
        cumsum += v
        if len(q) > k:
            cumsum -= q.pop(0)
        out.append(list((cumsum / len(q)).tolist()))
    return out

def face_boxes_from_yolo(res, min_face_size: int) -> List[Tuple[int,int,int,int,float]]:
    boxes = []
    for r in res:
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            if w >= min_face_size and h >= min_face_size:
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0])))
    return boxes

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "labels": EMO_LABELS,
        "models_path": str(MODELS),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    sample_every_s: float = Form(2.0),      # 1 output every N seconds
    conf: float = Form(0.25),               # YOLO confidence
    maxFaces: int = Form(1),                # 0 = all faces, else largest N faces
    return_boxes: int = Form(0),            # 1 to include face boxes in response
    smooth_k: int = Form(1),                # moving-average smoothing over windows
    topk: int = Form(3),                    # top-K emotions to include per-window/overall
    min_face_size: int = Form(20),          # ignore tiny boxes
):
    # Save temp file
    tmpdir = Path(tempfile.mkdtemp(dir=str(ROOT)))
    tmpvid = tmpdir / file.filename
    with open(tmpvid, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(str(tmpvid))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = float(total_frames / max(vid_fps, 1.0)) if total_frames > 0 else 0.0
    step = max(float(sample_every_s), 0.1)
    # floor => exactly N outputs: e.g., 20s // 2s = 10
    expected_windows = int(duration_s // step) if duration_s > 0 else None

    windows_raw_probs: List[List[float]] = []
    windows_faces: List[int] = []
    windows_time: List[float] = []
    windows_boxes: List[List[Tuple[int,int,int,int,float]]] = []

    next_boundary = 0.0
    frame_idx = 0
    win_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tsec = float(frame_idx / max(vid_fps, 1.0))
        frame_idx += 1

        if tsec + 1e-6 < next_boundary:
            continue  # not yet the next window

        # Representative frame for this window
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Detect faces
        res = yolo.predict(pil, imgsz=640, conf=conf, verbose=False)
        boxes = face_boxes_from_yolo(res, min_face_size=min_face_size)
        # keep largest N
        boxes_sorted = sorted(boxes, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        if maxFaces > 0:
            boxes_sorted = boxes_sorted[:maxFaces]

        # Emotion inference
        face_probs: List[List[float]] = []
        for (x1, y1, x2, y2, _c) in boxes_sorted:
            crop = pil.crop((x1, y1, x2, y2))
            x = emo_tf(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = emo_model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            face_probs.append(probs)

        # Aggregate faces (mean)
        if face_probs:
            avg_probs = list(np.mean(np.array(face_probs), axis=0).tolist())
        else:
            avg_probs = []  # no face in this window

        windows_raw_probs.append(avg_probs)
        windows_faces.append(len(boxes_sorted))
        windows_time.append(round(next_boundary, 2))
        if return_boxes:
            windows_boxes.append(boxes_sorted)

        win_idx += 1
        next_boundary += step

        if expected_windows is not None and win_idx >= expected_windows:
            break

    cap.release()
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Optionally smooth across windows
    smoothed_probs = moving_average(
        [p if p else [0.0] * len(EMO_LABELS) for p in windows_raw_probs],
        k=int(smooth_k)
    ) if len(windows_raw_probs) > 0 else []

    # Build detailed windows
    windows_out: List[Dict[str, Any]] = []
    for i, t0 in enumerate(windows_time):
        probs = smoothed_probs[i] if smoothed_probs else []
        faces = windows_faces[i]
        entropy = shannon_entropy(probs) if probs else 0.0
        stress_score = stress_score_from_probs(EMO_LABELS, probs) if probs else 0.0
        stress_flag = bool(stress_score >= STRESS_THRESHOLD)
        top = topk_from_probs(EMO_LABELS, probs, k=int(topk)) if probs else []

        item: Dict[str, Any] = {
            "window_index": i,
            "t_start": t0,
            "t_center": round(t0 + step / 2.0, 2),
            "faces": faces,
            "entropy": entropy,
            "emotion_probs": probs,   # aligned with EMO_LABELS
            "top_emotions": top,
            "stress_score": stress_score,
            "stress": stress_flag,
        }
        if return_boxes:
            # [ [x1,y1,x2,y2,conf], ... ] for this window
            item["boxes"] = windows_boxes[i] if i < len(windows_boxes) else []
        windows_out.append(item)

    # Overall stats
    valid_probs = [w["emotion_probs"] for w in windows_out if w["emotion_probs"]]
    if valid_probs:
        arr = np.array(valid_probs, dtype=np.float32)
        mean_probs = arr.mean(axis=0).tolist()
        std_probs  = arr.std(axis=0).tolist()
        min_probs  = arr.min(axis=0).tolist()
        max_probs  = arr.max(axis=0).tolist()
    else:
        mean_probs = std_probs = min_probs = max_probs = []

    overall_sorted = topk_from_probs(EMO_LABELS, mean_probs, k=int(topk)) if mean_probs else []
    overall_stress_score = float(np.mean([w["stress_score"] for w in windows_out])) if windows_out else 0.0
    overall_stress_flag  = bool(overall_stress_score >= STRESS_THRESHOLD)

    # Valence/Arousal (based on mean probs)
    if mean_probs:
        overall_valence, overall_arousal = valence_arousal_from_probs(EMO_LABELS, mean_probs)
    else:
        overall_valence, overall_arousal = 0.0, 0.0

    per_label_stats = []
    for i, lab in enumerate(EMO_LABELS):
        per_label_stats.append({
            "label": lab,
            "mean": float(mean_probs[i]) if mean_probs else 0.0,
            "std":  float(std_probs[i])  if std_probs  else 0.0,
            "min":  float(min_probs[i])  if min_probs  else 0.0,
            "max":  float(max_probs[i])  if max_probs  else 0.0,
        })

    resp = {
        "meta": {
            "device": DEVICE,
            "labels": EMO_LABELS,
            "video": {
                "fps": float(vid_fps),
                "total_frames": int(total_frames),
                "duration_s": float(duration_s),
            },
            "windowing": {
                "sample_every_s": float(step),
                "expected_windows": int(expected_windows) if expected_windows is not None else None,
                "returned_windows": len(windows_out),
                "smooth_k": int(smooth_k),
                "yolo_conf": float(conf),
                "maxFaces": int(maxFaces),
                "min_face_size": int(min_face_size),
            },
            "stress": {
                "weights": EMO_WEIGHTS,
                "threshold": STRESS_THRESHOLD,
            },
        },
        "windows": windows_out,
        "overall": {
            "top_emotions": overall_sorted,
            "per_label_stats": per_label_stats,
            "stress_score": overall_stress_score,
            "stress": overall_stress_flag,
            "valence": overall_valence,
            "arousal": overall_arousal,
        },
    }
    return resp

# Optional: quick image endpoint for single-frame tests
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    maxFaces: int = Form(1),
    return_boxes: int = Form(0),
    min_face_size: int = Form(20),
    topk: int = Form(3),
):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    res = yolo.predict(pil, imgsz=640, conf=conf, verbose=False)
    boxes = face_boxes_from_yolo(res, min_face_size=min_face_size)
    boxes_sorted = sorted(boxes, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
    if maxFaces > 0:
        boxes_sorted = boxes_sorted[:maxFaces]

    face_probs: List[List[float]] = []
    for (x1, y1, x2, y2, _c) in boxes_sorted:
        crop = pil.crop((x1, y1, x2, y2))
        x = emo_tf(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = emo_model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        face_probs.append(probs)

    if face_probs:
        avg_probs = list(np.mean(np.array(face_probs), axis=0).tolist())
        entropy = shannon_entropy(avg_probs)
        stress_score = stress_score_from_probs(EMO_LABELS, avg_probs)
        stress_flag = bool(stress_score >= STRESS_THRESHOLD)
        overall_sorted = topk_from_probs(EMO_LABELS, avg_probs, k=int(topk))
        val, aro = valence_arousal_from_probs(EMO_LABELS, avg_probs)
    else:
        avg_probs = []
        entropy = 0.0
        stress_score = 0.0
        stress_flag = False
        overall_sorted = []
        val, aro = 0.0, 0.0

    out: Dict[str, Any] = {
        "labels": EMO_LABELS,
        "faces": len(boxes_sorted),
        "entropy": entropy,
        "emotion_probs": avg_probs,
        "top_emotions": overall_sorted,
        "stress_score": stress_score,
        "stress": stress_flag,
        "valence": val,
        "arousal": aro,
    }
    if return_boxes:
        out["boxes"] = boxes_sorted
    return out

if __name__ == "__main__":
    # Optional local runner: adjust module path if you move this file
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
