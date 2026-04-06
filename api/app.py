from __future__ import annotations
import pickle
from pathlib import Path
from joblib import load
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, Body
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
import uuid

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

# ---------- Load keystroke RF model ----------
KF_MODEL_PATH = MODELS / "keystroke_rf_v1.pkl"
KF_FEATURES_PATH = MODELS / "keystroke_features_v1.pkl"

if not KF_MODEL_PATH.exists():
    print(f"⚠️ Keystroke RF model not found at {KF_MODEL_PATH} — /keystrokes route will be unavailable.")
    keystroke_model = None
    FEATURE_ORDER = []
else:
    keystroke_model = load(KF_MODEL_PATH)
    print("🔹 Loaded keystroke RandomForest model:", KF_MODEL_PATH.name)
    
    # Try to load feature names from pickle file, or use fallback
    if KF_FEATURES_PATH.exists():
        try:
            FEATURE_ORDER = load(KF_FEATURES_PATH)
            print(f"🔹 Loaded feature order from {KF_FEATURES_PATH.name}: {len(FEATURE_ORDER)} features")
        except Exception as e:
            print(f"⚠️ Could not load features from {KF_FEATURES_PATH}: {e}")
            FEATURE_ORDER = None
    else:
        FEATURE_ORDER = None
    
    # Check model's expected feature count
    if hasattr(keystroke_model, "n_features_"):
        expected_features = keystroke_model.n_features_
        print(f"🔹 Model expects {expected_features} features")
        if FEATURE_ORDER and len(FEATURE_ORDER) != expected_features:
            print(f"⚠️ WARNING: Feature order has {len(FEATURE_ORDER)} features but model expects {expected_features}")
    
    # Fallback feature order if not loaded from file
    if FEATURE_ORDER is None:
        print("⚠️ Using fallback feature order (may not match training)")
        FEATURE_ORDER = [
            "duration_ms","n_keydowns","n_keyups","chars_per_sec",
            "dwell_mean_ms","dwell_std_ms","dwell_median_ms","dwell_p10_ms","dwell_p90_ms",
            "dd_mean_ms","dd_std_ms","dd_median_ms",
            "pauses_gt_200","pauses_gt_500","pauses_gt_1000","longest_pause_ms",
            "backspace_count","cv_dwell","cv_dd",
            "hist_bin_0","hist_bin_1","hist_bin_2","hist_bin_3","hist_bin_4","hist_bin_5",
            "inter_entropy"
        ]

MODEL_VERSION = "rf_v1.0"

# ---------- Session Management ----------
# In-memory session store (for production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

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
    session_id: Optional[str] = Form(None), # Optional: store in session
    question_id: Optional[int] = Form(None), # Optional: question number
    question_text: Optional[str] = Form(None), # Optional: question text
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
    
    # Store in session if session_id and question_id provided
    if session_id and question_id is not None:
        if session_id not in sessions:
            sessions[session_id] = {
                "session_id": session_id,
                "user_id": "",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "questions": [],
                "completed": False,
            }
        
        # Find or create question entry
        question_idx = None
        for i, q in enumerate(sessions[session_id]["questions"]):
            if q.get("question_id") == question_id:
                question_idx = i
                break
        
        if question_idx is None:
            # Create new question entry
            sessions[session_id]["questions"].append({
                "question_id": question_id,
                "question_text": question_text if question_text and question_text.strip() else None,
                "video_analysis": resp,
                "keystroke_analysis": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            # Update existing question entry
            sessions[session_id]["questions"][question_idx]["video_analysis"] = resp
            if question_text and question_text.strip():
                sessions[session_id]["questions"][question_idx]["question_text"] = question_text
    
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

@app.post("/keystrokes")
async def predict_keystroke_stress(payload: Dict[str, Any] = Body(...)):
    """
    Keystroke Stress Detection endpoint.
    Expects JSON with 'features' object and some meta (session_id, user_id optional).
    """
    if keystroke_model is None:
        return {"ok": False, "error": "Keystroke model not loaded on server."}

    # Basic payload validation
    session_id = payload.get("session_id", "")
    user_id = payload.get("user_id", "")
    question_id = payload.get("question_id", None)  # Extract question_id for session storage
    features = payload.get("features")

    if not isinstance(features, dict):
        return {"ok": False, "error": "Missing or invalid 'features' field."}

    # Validate feature order length matches model
    if hasattr(keystroke_model, "n_features_"):
        expected_count = keystroke_model.n_features_
        if len(FEATURE_ORDER) != expected_count:
            return {
                "ok": False,
                "error": f"Feature order mismatch: FEATURE_ORDER has {len(FEATURE_ORDER)} features but model expects {expected_count}. Please check FEATURE_ORDER in app.py",
                "expected_features": expected_count,
                "provided_features": len(FEATURE_ORDER),
                "feature_order": FEATURE_ORDER
            }

    # Build input vector in the exact training order
    try:
        x = np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)
        
        # Validate shape before prediction
        if hasattr(keystroke_model, "n_features_") and x.shape[1] != keystroke_model.n_features_:
            return {
                "ok": False,
                "error": f"Feature count mismatch: generated {x.shape[1]} features but model expects {keystroke_model.n_features_}",
                "expected_features": keystroke_model.n_features_,
                "generated_features": int(x.shape[1]),
                "feature_order_length": len(FEATURE_ORDER)
            }
    except Exception as e:
        return {"ok": False, "error": f"Invalid feature values: {str(e)}"}

    # Run model
    try:
        # If model supports predict_proba
        if hasattr(keystroke_model, "predict_proba"):
            probs = keystroke_model.predict_proba(x)[0]  # shape = (n_classes,)
            classes = keystroke_model.classes_
            # compute predicted numeric class (as provided by model)
            pred_class = int(keystroke_model.predict(x)[0])
            confidence = float(max(probs))
            # Define stressed classes: if labels are 1-5, treat >=4 as stressed; else if 0/1...
            stressed_mask = [((c >= 4) if isinstance(c, (int, float)) else False) for c in classes]
            stress_score = float(np.sum([p for p, m in zip(probs, stressed_mask) if m]))
        else:
            pred_class = int(keystroke_model.predict(x)[0])
            probs = None
            confidence = 1.0
            # fallback: high stress if predicted >=4
            stress_score = 1.0 if pred_class >= 4 else (0.5 if pred_class == 3 else 0.0)

        # Map numeric class to label (flexible if your labels are 1-5 or 0-2)
        if pred_class >= 4:
            stress_label = "High"
            stress_class = 2
        elif pred_class == 3:
            stress_label = "Medium"
            stress_class = 1
        else:
            stress_label = "Low"
            stress_class = 0

        # Feature importances (top 5)
        fi = {}
        try:
            importances = getattr(keystroke_model, "feature_importances_", None)
            if importances is not None:
                idxs = np.argsort(importances)[::-1][:10]
                fi = {FEATURE_ORDER[int(i)]: float(round(importances[int(i)], 6)) for i in idxs}
        except Exception:
            fi = {}

        resp = {
            "ok": True,
            "session_id": session_id,
            "user_id": user_id,
            "stress_label": stress_label,
            "stress_score": round(float(stress_score), 4),
            "stress_class": int(stress_class),
            "predicted_class_raw": int(pred_class),
            "confidence": round(float(confidence), 4),
            "model_version": MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_importance": fi,
        }
        
        # Store in session if session_id and question_id provided
        if session_id and question_id is not None:
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": user_id or "",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "questions": [],
                    "completed": False,
                }
            
            # Find or create question entry
            question_idx = None
            for i, q in enumerate(sessions[session_id]["questions"]):
                if q.get("question_id") == question_id:
                    question_idx = i
                    break
            
            # Get question_text from payload if available
            question_text_from_payload = payload.get("question_text", "")
            
            if question_idx is None:
                # Create new question entry
                sessions[session_id]["questions"].append({
                    "question_id": question_id,
                    "question_text": question_text_from_payload if question_text_from_payload else None,
                    "video_analysis": None,
                    "keystroke_analysis": resp,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                # Update existing question entry
                sessions[session_id]["questions"][question_idx]["keystroke_analysis"] = resp
                if question_text_from_payload:
                    sessions[session_id]["questions"][question_idx]["question_text"] = question_text_from_payload
        
        # optionally echo back requested pieces for debugging:
        # resp["features_received"] = {k: features.get(k) for k in FEATURE_ORDER}
        return resp
    except Exception as e:
        return {"ok": False, "error": f"Model inference error: {str(e)}"}

@app.post("/start-session")
async def start_session(payload: Optional[Dict[str, Any]] = Body(None)):
    """
    Start a new session and generate a unique session_id.
    Returns the session_id and initializes an empty session.
    """
    session_id = str(uuid.uuid4())
    user_id = payload.get("user_id", "") if payload else ""
    
    sessions[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "questions": [],
        "completed": False,
    }
    
    return {
        "ok": True,
        "session_id": session_id,
        "user_id": user_id,
        "started_at": sessions[session_id]["started_at"],
    }

@app.post("/submit-question")
async def submit_question(
    session_id: str = Form(...),
    question_id: int = Form(...),
    question_text: str = Form(""),
    video: UploadFile = File(...),
    keystroke_features: str = Form(...),  # JSON string of keystroke features
    window_start_ts: int = Form(...),
    window_end_ts: int = Form(...),
    sample_every_s: float = Form(2.0),
    conf: float = Form(0.25),
    maxFaces: int = Form(1),
    return_boxes: int = Form(0),
    smooth_k: int = Form(1),
    topk: int = Form(3),
    min_face_size: int = Form(20),
):
    """
    Submit both video and keystroke data for a question.
    Analyzes both and stores the results in the session.
    Returns the combined response for this question.
    """
    # Verify session exists
    if session_id not in sessions:
        return {"ok": False, "error": f"Session {session_id} not found. Please start a session first."}
    
    # Parse keystroke features
    try:
        features_dict = json.loads(keystroke_features)
    except Exception as e:
        return {"ok": False, "error": f"Invalid keystroke_features JSON: {str(e)}"}
    
    # Analyze video
    video_analysis = None
    try:
        # Save temp file
        tmpdir = Path(tempfile.mkdtemp(dir=str(ROOT)))
        tmpvid = tmpdir / video.filename
        with open(tmpvid, "wb") as f:
            content = await video.read()
            f.write(content)
        
        cap = cv2.VideoCapture(str(tmpvid))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = float(total_frames / max(vid_fps, 1.0)) if total_frames > 0 else 0.0
        step = max(float(sample_every_s), 0.1)
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
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

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
            else:
                avg_probs = []

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

        smoothed_probs = moving_average(
            [p if p else [0.0] * len(EMO_LABELS) for p in windows_raw_probs],
            k=int(smooth_k)
        ) if len(windows_raw_probs) > 0 else []

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
                "emotion_probs": probs,
                "top_emotions": top,
                "stress_score": stress_score,
                "stress": stress_flag,
            }
            if return_boxes:
                item["boxes"] = windows_boxes[i] if i < len(windows_boxes) else []
            windows_out.append(item)

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

        video_analysis = {
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
    except Exception as e:
        return {"ok": False, "error": f"Video analysis failed: {str(e)}"}
    
    # Analyze keystroke
    keystroke_analysis = None
    if keystroke_model is not None:
        try:
            # Validate feature order length matches model
            if hasattr(keystroke_model, "n_features_"):
                expected_count = keystroke_model.n_features_
                if len(FEATURE_ORDER) != expected_count:
                    return {
                        "ok": False,
                        "error": f"Feature order mismatch: FEATURE_ORDER has {len(FEATURE_ORDER)} features but model expects {expected_count}",
                        "expected_features": expected_count,
                        "provided_features": len(FEATURE_ORDER),
                    }

            # Build input vector in the exact training order
            x = np.array([float(features_dict.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)
            
            if hasattr(keystroke_model, "n_features_") and x.shape[1] != keystroke_model.n_features_:
                return {
                    "ok": False,
                    "error": f"Feature count mismatch: generated {x.shape[1]} features but model expects {keystroke_model.n_features_}",
                }

            if hasattr(keystroke_model, "predict_proba"):
                probs = keystroke_model.predict_proba(x)[0]
                classes = keystroke_model.classes_
                pred_class = int(keystroke_model.predict(x)[0])
                confidence = float(max(probs))
                stressed_mask = [((c >= 4) if isinstance(c, (int, float)) else False) for c in classes]
                stress_score = float(np.sum([p for p, m in zip(probs, stressed_mask) if m]))
            else:
                pred_class = int(keystroke_model.predict(x)[0])
                probs = None
                confidence = 1.0
                stress_score = 1.0 if pred_class >= 4 else (0.5 if pred_class == 3 else 0.0)

            if pred_class >= 4:
                stress_label = "High"
                stress_class = 2
            elif pred_class == 3:
                stress_label = "Medium"
                stress_class = 1
            else:
                stress_label = "Low"
                stress_class = 0

            fi = {}
            try:
                importances = getattr(keystroke_model, "feature_importances_", None)
                if importances is not None:
                    idxs = np.argsort(importances)[::-1][:10]
                    fi = {FEATURE_ORDER[int(i)]: float(round(importances[int(i)], 6)) for i in idxs}
            except Exception:
                fi = {}

            keystroke_analysis = {
                "ok": True,
                "stress_label": stress_label,
                "stress_score": round(float(stress_score), 4),
                "stress_class": int(stress_class),
                "predicted_class_raw": int(pred_class),
                "confidence": round(float(confidence), 4),
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feature_importance": fi,
            }
        except Exception as e:
            keystroke_analysis = {"ok": False, "error": f"Keystroke analysis failed: {str(e)}"}
    else:
        keystroke_analysis = {"ok": False, "error": "Keystroke model not loaded on server."}
    
    # Store question response in session
    question_response = {
        "question_id": question_id,
        "question_text": question_text,
        "window_start_ts": window_start_ts,
        "window_end_ts": window_end_ts,
        "video_analysis": video_analysis,
        "keystroke_analysis": keystroke_analysis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    sessions[session_id]["questions"].append(question_response)
    
    return {
        "ok": True,
        "session_id": session_id,
        "question_id": question_id,
        "video_analysis": video_analysis,
        "keystroke_analysis": keystroke_analysis,
        "message": "Question submitted successfully",
    }

@app.get("/get-session/{session_id}")
async def get_session(session_id: str):
    """
    Get complete session data with all question responses.
    Returns in the format: {session_id, questions: [{question_id, video_analysis, keystroke_analysis}, ...]}
    """
    if session_id not in sessions:
        return {"ok": False, "error": f"Session {session_id} not found."}
    
    session = sessions[session_id]
    
    return {
        "ok": True,
        "session_id": session_id,
        "user_id": session.get("user_id", ""),
        "started_at": session["started_at"],
        "completed": session["completed"],
        "questions": session["questions"],
    }

@app.post("/complete-session/{session_id}")
async def complete_session(session_id: str):
    """
    Mark a session as completed.
    """
    if session_id not in sessions:
        return {"ok": False, "error": f"Session {session_id} not found."}
    
    sessions[session_id]["completed"] = True
    sessions[session_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session completed successfully",
    }

if __name__ == "__main__":
    # Optional local runner: adjust module path if you move this file
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
