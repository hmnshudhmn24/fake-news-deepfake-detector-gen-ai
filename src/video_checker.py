# src/video_checker.py
import os
import cv2
import numpy as np
from skimage import filters
from .models import predict_batch
from tempfile import mkdtemp
from typing import Dict, List

def extract_frames(video_path: str, fps: float = 0.5, out_dir: str = None) -> List[str]:
    if out_dir is None:
        out_dir = mkdtemp(prefix="frames_")
    os.makedirs(out_dir, exist_ok=True)

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        return []

    video_fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / max(0.001, fps))))
    saved = []
    idx = 0
    frame_i = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if idx % step == 0:
            p = os.path.join(out_dir, f"frame_{frame_i:06d}.jpg")
            cv2.imwrite(p, frame)
            saved.append(p)
            frame_i += 1
        idx += 1
    vid.release()
    return saved

def face_crop_from_image(image_path: str):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    crops = []
    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        crops.append(crop)
    return crops

def blur_score(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def color_banding_score(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges = filters.sobel(gray)
    return 1.0 - np.mean(edges)

def preprocess_for_model(crop_np, size=(224,224)):
    import torch
    from torchvision import transforms
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(pil)

def analyze_video(video_path: str, fps: float = 0.5) -> Dict:
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return {"error": "no_frames", "notes": "Could not open video or extract frames."}
    per_frame_scores = []
    batch_tensors = []
    batch_paths = []

    for fpath in frames:
        crops = face_crop_from_image(fpath)
        if not crops:
            frame_img = cv2.imread(fpath)
            crops = [frame_img]
        for crop in crops:
            blur = blur_score(crop)
            band = color_banding_score(crop)
            try:
                t = preprocess_for_model(crop)
                batch_tensors.append(t)
                batch_paths.append(fpath)
            except Exception:
                continue
            per_frame_scores.append({
                "frame": fpath,
                "blur": blur,
                "banding": float(band)
            })

    results = []
    if batch_tensors:
        import torch
        batch = torch.stack(batch_tensors)
        probs = predict_batch(batch)
        for i, p in enumerate(probs):
            results.append({"frame_path": batch_paths[i], "score": float(p)})
    else:
        results = []

    combined = []
    for r in results:
        heur = next((h for h in per_frame_scores if h["frame"] == r["frame_path"]), None)
        if heur:
            bnorm = max(0.0, min(1.0, 1.0 - min(heur["blur"] / 100.0, 1.0)))
            band = heur["banding"]
            combined_score = 0.6 * r["score"] + 0.25 * bnorm + 0.15 * band
        else:
            combined_score = r["score"]
        combined.append({"frame_path": r["frame_path"], "score": combined_score})

    combined_sorted = sorted(combined, key=lambda x: -x["score"])
    import numpy as _np
    fake_prob = float(_np.mean([c["score"] for c in combined_sorted])) if combined_sorted else 0.0

    return {
        "fake_prob": fake_prob,
        "top_frames": combined_sorted,
        "notes": "This is heuristic output. Train a classifier on DFDC/FaceForensics++ for production."
    }
