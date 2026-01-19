from __future__ import annotations
import os
import re
import sys
import json
import time
import math
import zipfile
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import confusion_matrix, classification_report
from dotenv import load_dotenv 
load_dotenv()
# --- OpenAI (real LLM) ---
from openai import OpenAI

_OPENAI_CLIENT: Optional[OpenAI] = None

def _get_openai_client() -> Optional[OpenAI]:
    global _OPENAI_CLIENT
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT

def llm_feedback_for_row_openai(true_label: str, pred_label: str, confidence: float, correct: bool) -> Dict[str, str]:
    """
    Uses OpenAI API (gpt-4o-mini) and returns:
    {"llm_keep": "...", "llm_improve": "..."}
    """
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing")

    system_msg = (
        "You are a strict, helpful squat-technique evaluator. "
        "Return concise coaching feedback in English only."
    )

    user_msg = f"""
Context:
- true_label: {true_label}
- pred_label: {pred_label}
- confidence: {confidence:.4f}
- correct: {correct}

Task:
Return JSON with:
- llm_keep: 2 short points separated by " ; "
- llm_improve: 2 short points separated by " ; "
Rules:
- English only
- No extra keys
- No markdown
"""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "llm_keep": {"type": "string"},
            "llm_improve": {"type": "string"},
        },
        "required": ["llm_keep", "llm_improve"],
    }

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "squat_feedback",
                "schema": schema,
                "strict": True,
            }
        },
    )

    data = json.loads(resp.output_text)
    return {"llm_keep": data["llm_keep"], "llm_improve": data["llm_improve"]}

# ============================================================
# 0) Dependency setup (Colab-friendly, with REAL validation)
# ============================================================
def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def _pip_install(pkgs: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs
    subprocess.check_call(cmd)

def _maybe_fix_shadowing(package_name: str):
    cwd = Path.cwd()
    candidates = [cwd / f"{package_name}.py", cwd / package_name]
    for cand in candidates:
        if cand.exists():
            new_name = cand.with_name(cand.name + "_LOCAL_BACKUP")
            print(f"Found local {cand} shadowing '{package_name}' -> renaming to {new_name}")
            try:
                cand.rename(new_name)
            except Exception as e:
                print(" Failed to rename shadowing file/folder:", e)

def ensure_deps():
    needed = [
        "mediapipe>=0.10.14",
        "diffusers>=0.29.0",
        "transformers>=4.41.0",
        "accelerate>=0.31.0",
        "safetensors>=0.4.3",
        "opencv-python",
        "pillow",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "torch",
        "torchvision",
    ]

    _maybe_fix_shadowing("mediapipe")

    try:
        import mediapipe as mp  
        ok_mp = hasattr(mp, "solutions")
        if not ok_mp:
            raise RuntimeError("mediapipe import exists but mp.solutions is missing (broken install or shadowed).")
        import diffusers  
        import transformers  
        import accelerate  
        import cv2  
        import PIL  
        import numpy  
        import pandas 
        import sklearn 
        import matplotlib  
        import tqdm 
        import torch  
        import torchvision  
    except Exception as e:
        print("Installing/repairing dependencies...")
        print("Reason:", repr(e))
        _pip_install(needed)
        import importlib
        importlib.invalidate_caches()

ensure_deps()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn

from PIL import Image

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from diffusers import ControlNetModel
try:
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
except Exception:
    from diffusers.pipelines.controlnet import StableDiffusionControlNetImg2ImgPipeline  # type: ignore
from diffusers import DPMSolverMultistepScheduler

# ============================================================
# 1) Config + ENV overrides
# ============================================================
def _env_int(name: str, default: int) -> int:
    s = os.environ.get(name, "").strip()
    return int(s) if s else default

def _env_float(name: str, default: float) -> float:
    s = os.environ.get(name, "").strip()
    return float(s) if s else default

@dataclass
class CFG:

    TOTAL_IMAGES: int = 2000
    VAL_RATIO: float = 0.2
    TEST_RATIO: float = 0.0
    IMG_SIZE: int = 512
    CONTROL_SIZE: int = 512
    CANVAS_MARGIN: float = 0.08

    BATCH_SIZE: int = 16
    WARMUP_EPOCHS: int = 1
    FINETUNE_EPOCHS: int = 8
    UNFREEZE_LAST_BLOCKS: int = 4

    LR_WARMUP: float = 2e-4
    LR_FINETUNE: float = 3e-5
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05
    GRAD_ACCUM_STEPS: int = 1
    EARLY_STOP_PATIENCE: int = 3

    MIXUP_ALPHA: float = 0.0  

    SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_MODEL: str = "lllyasviel/sd-controlnet-openpose"

    PROMPT_GOOD: str = (
        "Realistic photo of a person performing a perfect bodyweight squat in a modern gym, "
        "heels flat, knees tracking over toes, neutral spine, chest up, controlled depth, sharp focus, natural lighting."
    )
    PROMPT_BAD: str = (
        "Realistic photo of a person performing a squat with bad form in a modern gym, "
        "knees collapsing inward, heels lifting, excessive forward torso lean, rounded lower back, unstable posture, sharp focus."
    )
    NEGATIVE_PROMPT: str = (
        "cartoon, illustration, anime, low quality, blurry, deformed, extra limbs, extra fingers, "
        "bad anatomy, bad proportions, distorted face, watermark, text, logo"
    )

    IMG2IMG_STRENGTH_GOOD: float = 0.65
    IMG2IMG_STRENGTH_BAD: float = 0.75

    NUM_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    CONTROLNET_SCALE: float = 1.0

    SYNTH_BAD_FROM_BADSEEDS: bool = True
    BAD_FROM_BADSEEDS_FRACTION_TRAIN: float = 0.25

    SEED: int = 42
    ZIP_NAME: str = "PoseAITraining_artifact.zip"

cfg = CFG()

cfg.TOTAL_IMAGES = _env_int("TOTAL_IMAGES", cfg.TOTAL_IMAGES)
cfg.FINETUNE_EPOCHS = _env_int("FINETUNE_EPOCHS", cfg.FINETUNE_EPOCHS)
cfg.WARMUP_EPOCHS = _env_int("WARMUP_EPOCHS", cfg.WARMUP_EPOCHS)
cfg.UNFREEZE_LAST_BLOCKS = _env_int("UNFREEZE_LAST_BLOCKS", cfg.UNFREEZE_LAST_BLOCKS)
cfg.BATCH_SIZE = _env_int("BATCH_SIZE", cfg.BATCH_SIZE)
cfg.LR_WARMUP = _env_float("LR_WARMUP", cfg.LR_WARMUP)
cfg.LR_FINETUNE = _env_float("LR_FINETUNE", cfg.LR_FINETUNE)
cfg.LABEL_SMOOTHING = _env_float("LABEL_SMOOTHING", cfg.LABEL_SMOOTHING)
cfg.GRAD_ACCUM_STEPS = max(1, _env_int("GRAD_ACCUM_STEPS", cfg.GRAD_ACCUM_STEPS))
cfg.MIXUP_ALPHA = _env_float("MIXUP_ALPHA", cfg.MIXUP_ALPHA)
cfg.EARLY_STOP_PATIENCE = _env_int("EARLY_STOP_PATIENCE", cfg.EARLY_STOP_PATIENCE)

# ============================================================
# 2) Paths
# ============================================================
ROOT = Path(".").resolve()

SEEDS_DIR = ROOT / "seeds"
GOOD_SEEDS_DIR = SEEDS_DIR / "good"
BAD_SEEDS_DIR = SEEDS_DIR / "bad"

SYNTH_DIR = ROOT / "synthetic_dataset"
TRAIN_DIR = SYNTH_DIR / "train"
VAL_DIR = SYNTH_DIR / "val"
TEST_DIR = SYNTH_DIR / "test"

OUTPUT_DIR = ROOT / "outputs"

for d in [
    GOOD_SEEDS_DIR, BAD_SEEDS_DIR,
    TRAIN_DIR / "good", TRAIN_DIR / "bad",
    VAL_DIR / "good", VAL_DIR / "bad",
    TEST_DIR / "good", TEST_DIR / "bad",
    OUTPUT_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ============================================================
# Minimal outputs mode
# ============================================================
MINIMAL_OUTPUT = os.environ.get("MINIMAL_OUTPUT", "1") == "1"

def allowed_output_files() -> set:
    return {
        "val_predictions.csv",
        "confidence_dist_val.png",
        "confusion_val.png",
        "roc_curve_val.png",
        "README.md",
    }

def prune_outputs_dir():
    if not OUTPUT_DIR.exists():
        return
    keep = allowed_output_files()
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and p.name not in keep:
            try:
                p.unlink()
            except Exception:
                pass

# ============================================================
# 3) Utilities
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

seed_everything(cfg.SEED)

def list_images(p: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not p.exists():
        return []
    out: List[Path] = []
    for q in p.iterdir():
        if q.is_file() and q.suffix.lower() in exts:
            out.append(q)
    return sorted(out)

def safe_imread_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def pil_center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    return img.resize((size, size), Image.BICUBIC)

def now_id() -> str:
    return str(int(time.time() * 1000))

def count_images(dirp: Path) -> int:
    return len(list_images(dirp))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def num_workers_auto() -> int:
    return 0 if os.name == "nt" else 2

# ============================================================
# 4) MediaPipe 3D pose extraction
# ============================================================
mp_pose = mp.solutions.pose

def extract_world_landmarks_from_image(pil_img: Image.Image) -> Optional[landmark_pb2.LandmarkList]:
    img = np.array(pil_img)  # RGB
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        res = pose.process(img)
        return res.pose_world_landmarks

def _lm_vec(lms, idx: int) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

# ============================================================
# 5) Skeleton tree + FK + 3D dragging
# ============================================================
class SkeletonNode:
    def __init__(self, name: str, parent: Optional["SkeletonNode"]=None, landmark_idx: Optional[int]=None):
        self.name = name
        self.parent = parent
        self.children: List["SkeletonNode"] = []
        self.landmark_idx = landmark_idx
        self.local_pos = np.zeros(3, dtype=np.float32)
        self.world_pos = np.zeros(3, dtype=np.float32)
        if parent is not None:
            parent.children.append(self)

def build_skeleton_tree(world_landmarks) -> Tuple[SkeletonNode, Dict[str, SkeletonNode], Dict[int, np.ndarray]]:
    lms = world_landmarks.landmark

    LH = mp_pose.PoseLandmark.LEFT_HIP.value
    RH = mp_pose.PoseLandmark.RIGHT_HIP.value
    LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    mid_hip = (_lm_vec(lms, LH) + _lm_vec(lms, RH)) / 2.0
    mid_shoulder = (_lm_vec(lms, LS) + _lm_vec(lms, RS)) / 2.0

    base_pts: Dict[int, np.ndarray] = {}
    for i in range(len(lms)):
        base_pts[i] = (_lm_vec(lms, i) - mid_hip).astype(np.float32)

    def get_centered(idx: int) -> np.ndarray:
        return base_pts[idx]

    root = SkeletonNode("MID_HIP", None, None)
    root.world_pos = np.zeros(3, dtype=np.float32)

    spine = SkeletonNode("MID_SHOULDER", root, None)
    spine.world_pos = (mid_shoulder - mid_hip).astype(np.float32)

    node_map: Dict[str, SkeletonNode] = {"MID_HIP": root, "MID_SHOULDER": spine}

    # legs
    l_hip = SkeletonNode("LEFT_HIP", root, LH);  l_hip.world_pos = get_centered(LH)
    l_knee = SkeletonNode("LEFT_KNEE", l_hip, mp_pose.PoseLandmark.LEFT_KNEE.value); l_knee.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_KNEE.value)
    l_ankle = SkeletonNode("LEFT_ANKLE", l_knee, mp_pose.PoseLandmark.LEFT_ANKLE.value); l_ankle.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_ANKLE.value)

    r_hip = SkeletonNode("RIGHT_HIP", root, RH); r_hip.world_pos = get_centered(RH)
    r_knee = SkeletonNode("RIGHT_KNEE", r_hip, mp_pose.PoseLandmark.RIGHT_KNEE.value); r_knee.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle = SkeletonNode("RIGHT_ANKLE", r_knee, mp_pose.PoseLandmark.RIGHT_ANKLE.value); r_ankle.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # arms
    l_sh = SkeletonNode("LEFT_SHOULDER", spine, LS); l_sh.world_pos = get_centered(LS)
    r_sh = SkeletonNode("RIGHT_SHOULDER", spine, RS); r_sh.world_pos = get_centered(RS)
    l_el = SkeletonNode("LEFT_ELBOW", l_sh, mp_pose.PoseLandmark.LEFT_ELBOW.value); l_el.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    r_el = SkeletonNode("RIGHT_ELBOW", r_sh, mp_pose.PoseLandmark.RIGHT_ELBOW.value); r_el.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    l_wr = SkeletonNode("LEFT_WRIST", l_el, mp_pose.PoseLandmark.LEFT_WRIST.value); l_wr.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_WRIST.value)
    r_wr = SkeletonNode("RIGHT_WRIST", r_el, mp_pose.PoseLandmark.RIGHT_WRIST.value); r_wr.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    for n in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle, l_sh, r_sh, l_el, r_el, l_wr, r_wr]:
        node_map[n.name] = n

    def compute_local(node: SkeletonNode):
        for ch in node.children:
            ch.local_pos = (ch.world_pos - node.world_pos).astype(np.float32)
            compute_local(ch)
    compute_local(root)

    def update_world(node: SkeletonNode):
        for ch in node.children:
            ch.world_pos = node.world_pos + ch.local_pos
            update_world(ch)
    update_world(root)

    return root, node_map, base_pts

def clone_skeleton(root: SkeletonNode, base_pts: Dict[int, np.ndarray]) -> Tuple[SkeletonNode, Dict[str, SkeletonNode], Dict[int, np.ndarray]]:
    node_map: Dict[str, SkeletonNode] = {}
    def _clone(node: SkeletonNode, parent: Optional[SkeletonNode]) -> SkeletonNode:
        nn_ = SkeletonNode(node.name, parent, node.landmark_idx)
        nn_.local_pos = node.local_pos.copy()
        nn_.world_pos = node.world_pos.copy()
        node_map[nn_.name] = nn_
        for ch in node.children:
            _clone(ch, nn_)
        return nn_
    new_root = _clone(root, None)
    new_base = {k: v.copy() for k, v in base_pts.items()}
    return new_root, node_map, new_base

def update_world_positions(root: SkeletonNode):
    def _upd(node: SkeletonNode):
        for ch in node.children:
            ch.world_pos = node.world_pos + ch.local_pos
            _upd(ch)
    _upd(root)

def skeleton_to_landmark_dict(root: SkeletonNode) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    def _walk(n: SkeletonNode):
        if n.landmark_idx is not None:
            out[int(n.landmark_idx)] = n.world_pos.copy()
        for c in n.children:
            _walk(c)
    _walk(root)
    return out

def merge_points(base_pts: Dict[int, np.ndarray], overrides: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    out = {k: v.copy() for k, v in base_pts.items()}
    for k, v in overrides.items():
        out[int(k)] = v.copy()
    return out

def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1,0,0],[0, math.cos(a), -math.sin(a)],[0, math.sin(a), math.cos(a)]], dtype=np.float32)

def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a),0, math.sin(a)],[0,1,0],[-math.sin(a),0, math.cos(a)]], dtype=np.float32)

def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a),0],[math.sin(a), math.cos(a),0],[0,0,1]], dtype=np.float32)

def apply_rotation_to_subtree(joint: SkeletonNode, R: np.ndarray):
    for ch in joint.children:
        ch.local_pos = (R @ ch.local_pos).astype(np.float32)
        apply_rotation_to_subtree(ch, R)

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-8)

def drag_joint_keep_bone_length(node: SkeletonNode, target_world: np.ndarray):
    if node.parent is None:
        return
    parent = node.parent
    L = _norm(node.local_pos)
    dir_vec = (target_world - parent.world_pos).astype(np.float32)
    d = _norm(dir_vec)
    if d < 1e-6:
        return
    new_dir = (dir_vec / d).astype(np.float32)
    node.local_pos = (new_dir * L).astype(np.float32)

def make_good_pose_jitter(root: SkeletonNode, node_map: Dict[str, SkeletonNode]):
    def rng(a, b): return random.uniform(a, b)
    if "MID_SHOULDER" in node_map and random.random() < 0.7:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_y(rng(-6, 6)))
    if "LEFT_HIP" in node_map and random.random() < 0.3:
        apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_z(rng(-4, 4)))
    if "RIGHT_HIP" in node_map and random.random() < 0.3:
        apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_z(rng(-4, 4)))
    update_world_positions(root)

def make_bad_pose_3d(root: SkeletonNode, node_map: Dict[str, SkeletonNode]) -> List[str]:
    def rng(a, b): return random.uniform(a, b)
    faults: List[str] = []

    if random.random() < 0.85: faults.append("forward_lean")
    if random.random() < 0.75: faults.append("knee_valgus")
    if random.random() < 0.50: faults.append("heel_lift")
    if random.random() < 0.40: faults.append("hip_shift")
    if random.random() < 0.35: faults.append("torso_twist")
    if random.random() < 0.25: faults.append("depth_shallow")

    if "forward_lean" in faults and "MID_SHOULDER" in node_map:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(20, 45)))

    if "torso_twist" in faults and "MID_SHOULDER" in node_map:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_y(rng(-18, 18)))
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(rng(-10, 10)))

    if "knee_valgus" in faults:
        for kn in ["LEFT_KNEE", "RIGHT_KNEE"]:
            if kn in node_map:
                k = node_map[kn]
                target = k.world_pos.copy()
                target[0] *= rng(0.35, 0.65)
                target[2] += rng(-0.05, 0.05)
                drag_joint_keep_bone_length(k, target)

    if "heel_lift" in faults:
        for an in ["LEFT_ANKLE", "RIGHT_ANKLE"]:
            if an in node_map:
                a = node_map[an]
                target = a.world_pos.copy()
                target[1] += rng(0.08, 0.18)
                target[2] += rng(-0.05, 0.10)
                drag_joint_keep_bone_length(a, target)

    if "hip_shift" in faults:
        shift = rng(-0.12, 0.12)
        for hn in ["LEFT_HIP", "RIGHT_HIP"]:
            if hn in node_map:
                h = node_map[hn]
                target = h.world_pos.copy()
                target[0] += shift * (1.0 if hn == "LEFT_HIP" else -1.0)
                drag_joint_keep_bone_length(h, target)

    if "depth_shallow" in faults:
        for kn in ["LEFT_KNEE", "RIGHT_KNEE"]:
            if kn in node_map:
                k = node_map[kn]
                target = k.world_pos.copy()
                target[1] += rng(0.05, 0.12)
                drag_joint_keep_bone_length(k, target)

    update_world_positions(root)
    return faults

# ============================================================
# 6) Camera sampling + projection + OpenPose-like control map
# ============================================================
def sample_camera_params() -> Dict[str, float]:
    return {
        "yaw": random.uniform(-50, 50),
        "pitch": random.uniform(-15, 15),
        "roll": random.uniform(-8, 8),
        "distance": random.uniform(3.0, 5.5),
        "fov": random.uniform(28, 55),
    }

def project_3d_to_2d(points_3d: Dict[int, np.ndarray], cam: Dict[str, float],
                     width: int, height: int, margin: float) -> Dict[int, Dict[str, float]]:
    y = math.radians(cam["yaw"])
    p = math.radians(cam["pitch"])
    r = math.radians(cam["roll"])

    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], dtype=np.float32)
    Rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]], dtype=np.float32)
    Rr = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    R = (Ry @ Rp @ Rr).astype(np.float32)

    cam_pos = (R @ np.array([0, 0, cam["distance"]], dtype=np.float32)).astype(np.float32)
    f = (height / 2.0) / math.tan(math.radians(cam["fov"]) / 2.0)

    pts2d: Dict[int, Tuple[float, float]] = {}
    xs, ys = [], []
    for idx, pt in points_3d.items():
        rel = (pt - cam_pos).astype(np.float32)
        cam_pt = (R.T @ rel).astype(np.float32)
        z = -float(cam_pt[2])
        if z < 0.1:
            z = 0.1
        px = (float(cam_pt[0]) * f / z) + (width / 2.0)
        py = (-float(cam_pt[1]) * f / z) + (height / 2.0)
        xs.append(px); ys.append(py)
        pts2d[idx] = (px, py)

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(1.0, maxx - minx)
    spany = max(1.0, maxy - miny)

    inner_w = width * (1.0 - 2.0 * margin)
    inner_h = height * (1.0 - 2.0 * margin)
    s = min(inner_w / spanx, inner_h / spany)

    out: Dict[int, Dict[str, float]] = {}
    for idx, (px, py) in pts2d.items():
        nx = (px - minx) * s + width * margin
        ny = (py - miny) * s + height * margin
        out[idx] = {"x": clamp01(nx / width), "y": clamp01(ny / height)}
    return out

def draw_openpose_like(projected: Dict[int, Dict[str, float]], width: int, height: int) -> Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for a, b in mp_pose.POSE_CONNECTIONS:
        if a in projected and b in projected:
            p1 = projected[a]; p2 = projected[b]
            x1, y1 = int(p1["x"] * width), int(p1["y"] * height)
            x2, y2 = int(p2["x"] * width), int(p2["y"] * height)
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 4)
    for _, p in projected.items():
        x, y = int(p["x"] * width), int(p["y"] * height)
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
    return Image.fromarray(canvas)

# ============================================================
# 7) Stage 1: SD Img2Img + ControlNet generation
# ============================================================
def load_pipe() -> StableDiffusionControlNetImg2ImgPipeline:
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    controlnet = ControlNetModel.from_pretrained(
        cfg.CONTROLNET_MODEL,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg.SD_MODEL,
        controlnet=controlnet,
        torch_dtype=TORCH_DTYPE,
        safety_checker=None,
        feature_extractor=None,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def _save_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)

def plan_targets(total_images: int) -> Dict[str, Dict[str, int]]:
    total_good = total_images // 2
    total_bad = total_images - total_good

    test_ratio = max(0.0, float(cfg.TEST_RATIO))
    val_ratio = max(0.0, float(cfg.VAL_RATIO))
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)

    def alloc(n: int) -> Tuple[int, int, int]:
        tr = int(round(n * train_ratio))
        va = int(round(n * val_ratio))
        te = n - tr - va
        return tr, va, te

    tr_g, va_g, te_g = alloc(total_good)
    tr_b, va_b, te_b = alloc(total_bad)

    return {
        "train": {"good": tr_g, "bad": tr_b},
        "val":   {"good": va_g, "bad": va_b},
        "test":  {"good": te_g, "bad": te_b},
    }

def generate_synthetic_dataset(total_images: int):
    print(f"\n=== Stage 1: Synthetic dataset (3D pose → manipulate → ControlNet Img2Img) on {DEVICE} ===")

    if DEVICE != "cuda" and os.environ.get("ALLOW_CPU_GENERATION", "0") != "1":
        print("Generation requires CUDA by default. Set ALLOW_CPU_GENERATION=1 to force CPU (slow).")
        return

    good_seeds = list_images(GOOD_SEEDS_DIR)
    bad_seeds = list_images(BAD_SEEDS_DIR)

    if not good_seeds:
        raise FileNotFoundError("No GOOD seeds found. Put images into seeds/good")

    random.shuffle(good_seeds)
    random.shuffle(bad_seeds)

    n_good = len(good_seeds)
    n_val = max(1, int(round(n_good * cfg.VAL_RATIO))) if n_good >= 2 else 0
    val_good_seeds = good_seeds[-n_val:] if n_val > 0 else []
    train_good_seeds = good_seeds[:-n_val] if n_val > 0 else good_seeds[:]

    targets = plan_targets(total_images)

    existing = {
        sp: {
            "good": count_images((SYNTH_DIR / sp / "good")),
            "bad":  count_images((SYNTH_DIR / sp / "bad")),
        } for sp in ["train", "val", "test"]
    }
    print("Existing counts:", existing)
    print("Target counts:  ", targets)

    pipe = load_pipe()

    pose_cache: Dict[str, Tuple[SkeletonNode, Dict[str, SkeletonNode], Image.Image, Dict[int, np.ndarray]]] = {}

    def get_pose_and_init(seed_path: Path) -> Optional[Tuple[SkeletonNode, Dict[str, SkeletonNode], Image.Image, Dict[int, np.ndarray]]]:
        key = str(seed_path.resolve())
        if key in pose_cache:
            return pose_cache[key]
        init_img = pil_center_crop_resize(safe_imread_pil(seed_path), cfg.IMG_SIZE)
        wlms = extract_world_landmarks_from_image(init_img)
        if wlms is None:
            return None
        root, node_map, base_pts = build_skeleton_tree(wlms)
        pose_cache[key] = (root, node_map, init_img, base_pts)
        return pose_cache[key]

    def rr(seeds: List[Path]):
        i = 0
        while True:
            yield seeds[i % len(seeds)] if seeds else None
            i += 1

    rr_train_good = rr(train_good_seeds)
    rr_val_good   = rr(val_good_seeds if val_good_seeds else train_good_seeds)
    rr_bad        = rr(bad_seeds)

    def gen_one(split: str, label: str, seed_path: Path, source: str) -> bool:
        pack = get_pose_and_init(seed_path)
        if pack is None:
            return False

        root_rest, _node_map_rest, init_img, base_pts_rest = pack
        cam = sample_camera_params()

        root, node_map, base_pts = clone_skeleton(root_rest, base_pts_rest)

        faults: List[str] = []
        if label == "good":
            if random.random() < 0.7:
                make_good_pose_jitter(root, node_map)
        else:
            if source == "bad_from_good":
                faults = make_bad_pose_3d(root, node_map)

        overrides = skeleton_to_landmark_dict(root)
        pts3d = merge_points(base_pts, overrides)

        proj = project_3d_to_2d(pts3d, cam, cfg.CONTROL_SIZE, cfg.CONTROL_SIZE, cfg.CANVAS_MARGIN)
        ctrl_img = draw_openpose_like(proj, cfg.CONTROL_SIZE, cfg.CONTROL_SIZE)

        if label == "good":
            prompt = cfg.PROMPT_GOOD
            strength = cfg.IMG2IMG_STRENGTH_GOOD
        else:
            prompt = cfg.PROMPT_BAD
            strength = cfg.IMG2IMG_STRENGTH_BAD

        g = torch.Generator(device=DEVICE)
        g.manual_seed(random.randint(0, 10**9))

        if DEVICE == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=TORCH_DTYPE)
        else:
            class _NoCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _NoCtx()

        with ctx:
            out = pipe(
                prompt=prompt,
                negative_prompt=cfg.NEGATIVE_PROMPT,
                image=init_img,
                control_image=ctrl_img,
                strength=strength,
                num_inference_steps=cfg.NUM_STEPS,
                guidance_scale=cfg.GUIDANCE_SCALE,
                controlnet_conditioning_scale=cfg.CONTROLNET_SCALE,
                generator=g
            ).images[0]

        fault_tag = ("__" + "__".join(faults[:3])) if faults else ""
        fname = f"synth_{label}_{source}{fault_tag}_{seed_path.stem}_{now_id()}.jpg"
        out_path = SYNTH_DIR / split / label / fname
        _save_image(out, out_path)

        meta = {
            "label": label,
            "source": source,
            "faults": faults,
            "seed": str(seed_path.name),
            "camera": cam,
            "prompt": prompt,
        }
        (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True

    def fill_split(split: str, label: str, target: int, seed_stream, source: str):
        while count_images(SYNTH_DIR / split / label) < target:
            seed_path = next(seed_stream)
            if seed_path is None:
                break
            ok = gen_one(split, label, seed_path, source=source)
            if not ok:
                continue

    fill_split("train", "good", targets["train"]["good"], rr_train_good, source="good_from_goodseeds")
    fill_split("val",   "good", targets["val"]["good"],   rr_val_good,   source="good_from_goodseeds")

    train_bad_target = targets["train"]["bad"]
    train_bad_from_badseeds = 0
    if cfg.SYNTH_BAD_FROM_BADSEEDS and bad_seeds and train_bad_target > 0:
        train_bad_from_badseeds = int(round(train_bad_target * cfg.BAD_FROM_BADSEEDS_FRACTION_TRAIN))
        train_bad_from_badseeds = max(0, min(train_bad_from_badseeds, train_bad_target))
    train_bad_from_good = train_bad_target - train_bad_from_badseeds

    fill_split("train", "bad", train_bad_from_good, rr_train_good, source="bad_from_good")
    fill_split("val",   "bad", targets["val"]["bad"],     rr_val_good,   source="bad_from_good")

    if train_bad_from_badseeds > 0:
        fill_split("train", "bad", train_bad_target, rr_bad, source="extra_bad_from_badseeds")

    print("\n Generation complete. Final counts:", {
        sp: {"good": count_images(SYNTH_DIR/sp/"good"), "bad": count_images(SYNTH_DIR/sp/"bad")}
        for sp in ["train", "val", "test"]
    })

# ============================================================
# 8) Dataset & Transforms
# ============================================================
class SquatDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        for label_name, y in [("bad", 0), ("good", 1)]:
            for p in list_images(self.root_dir / label_name):
                self.samples.append((p, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = safe_imread_pil(p)
        if self.transform:
            img = self.transform(img)
        return img, y, p.name

def _make_transforms():
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
    std  = weights.meta.get("std",  (0.229, 0.224, 0.225))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf

# ============================================================
# 9) Training
# ============================================================
def _unfreeze_last_blocks(model: nn.Module, n_blocks: int):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.parameters():
        p.requires_grad = True

    n_blocks = max(0, int(n_blocks))
    if n_blocks > 0 and hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        layers = model.encoder.layers
        for blk in list(layers)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(model.encoder, "ln"):
            for p in model.encoder.ln.parameters():
                p.requires_grad = True

def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())

def _soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    return -(target_probs * logp).sum(dim=1).mean()

def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        y_oh = torch.zeros((y.size(0), 2), device=y.device, dtype=torch.float32)
        y_oh.scatter_(1, y.view(-1, 1), 1.0)
        return x, y_oh

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]

    x_mix = lam * x + (1 - lam) * x2
    y_oh = torch.zeros((y.size(0), 2), device=y.device, dtype=torch.float32)
    y_oh.scatter_(1, y.view(-1, 1), 1.0)
    y2_oh = torch.zeros((y2.size(0), 2), device=y.device, dtype=torch.float32)
    y2_oh.scatter_(1, y2.view(-1, 1), 1.0)

    y_mix = lam * y_oh + (1 - lam) * y2_oh
    return x_mix, y_mix

def train_vit() -> Tuple[Optional[nn.Module], Optional[transforms.Compose], Dict[str, Any]]:
    print("\n=== Stage 2: Train ViT-B/16 on synthetic data (no extra output files) ===")

    train_tf, eval_tf = _make_transforms()

    train_ds = SquatDataset(TRAIN_DIR, transform=train_tf)
    val_ds   = SquatDataset(VAL_DIR, transform=eval_tf)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(" Missing training/val data in synthetic_dataset/.")
        return None, None, {}

    ys = [y for _, y in train_ds.samples]
    n0 = max(1, sum(1 for y in ys if y == 0))
    n1 = max(1, sum(1 for y in ys if y == 1))
    w0 = 1.0 / n0
    w1 = 1.0 / n1
    sample_w = [w0 if y == 0 else w1 for y in ys]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model.to(DEVICE)

    _unfreeze_last_blocks(model, n_blocks=0)

    num_workers = num_workers_auto()
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR_WARMUP,
        weight_decay=cfg.WEIGHT_DECAY
    )

    ce = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    def set_lr(lr: float):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    def eval_val() -> Tuple[float, float]:
        model.eval()
        losses = []
        accs = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE) if torch.is_tensor(y) else torch.tensor(y, device=DEVICE)
                logits = model(x)
                loss = ce(logits, y)
                losses.append(float(loss.item()))
                accs.append(_accuracy_from_logits(logits, y))
        return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0

    best_val_acc = -1.0
    no_improve = 0
    best_state_dict = None

    # --- Warmup ---
    for epoch in range(cfg.WARMUP_EPOCHS):
        model.train()
        set_lr(cfg.LR_WARMUP)

        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE) if torch.is_tensor(y) else torch.tensor(y, device=DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = ce(logits, y) / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * cfg.GRAD_ACCUM_STEPS
            running_acc += _accuracy_from_logits(logits.detach(), y)
            n_batches += 1

        tr_loss = running_loss / max(1, n_batches)
        tr_acc = running_acc / max(1, n_batches)
        va_loss, va_acc = eval_val()

        print(f"[Warmup {epoch+1}/{cfg.WARMUP_EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

    # --- Finetune ---
    _unfreeze_last_blocks(model, n_blocks=cfg.UNFREEZE_LAST_BLOCKS)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR_FINETUNE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    for epoch in range(cfg.FINETUNE_EPOCHS):
        model.train()

        t = epoch / max(1, cfg.FINETUNE_EPOCHS - 1)
        lr = cfg.LR_FINETUNE * (0.5 * (1.0 + math.cos(math.pi * t)))
        set_lr(lr)

        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE) if torch.is_tensor(y) else torch.tensor(y, device=DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                if cfg.MIXUP_ALPHA > 0:
                    x_mix, y_soft = _mixup_batch(x, y, cfg.MIXUP_ALPHA)
                    logits = model(x_mix)
                    loss = _soft_cross_entropy(logits, y_soft) / cfg.GRAD_ACCUM_STEPS
                    acc = _accuracy_from_logits(logits.detach(), torch.argmax(y_soft, dim=1))
                else:
                    logits = model(x)
                    loss = ce(logits, y) / cfg.GRAD_ACCUM_STEPS
                    acc = _accuracy_from_logits(logits.detach(), y)

            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * cfg.GRAD_ACCUM_STEPS
            running_acc += acc
            n_batches += 1

        tr_loss = running_loss / max(1, n_batches)
        tr_acc = running_acc / max(1, n_batches)
        va_loss, va_acc = eval_val()

        print(f"[Finetune {epoch+1}/{cfg.FINETUNE_EPOCHS}] lr {lr:.6f} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"Early stopping: no val improvement for {cfg.EARLY_STOP_PATIENCE} epochs.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"Training complete. Best val acc={best_val_acc:.4f}")
    return model, eval_tf, {"best_val_acc": best_val_acc}



# ============================================================
# 10) Evaluation + plots + LLM-like feedback columns
# ============================================================
def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, title: str):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _english_feedback_templates():
    keep_good = [
        "Heels stay grounded throughout the movement",
        "Knees track in line with the toes (no inward collapse)",
        "Neutral spine with chest up",
        "Controlled tempo on the way down and up",
    ]
    improve_good = [
        "Keep knees from collapsing inward (valgus control)",
        "Maintain flat feet (avoid heel lift)",
        "Keep a neutral lower back (avoid rounding)",
        "Maintain controlled depth without losing balance",
        "Reduce excessive forward torso lean",
    ]

    keep_bad = [
        "Move in a controlled tempo (avoid rushing)",
        "Keep gaze forward with chest up",
        "Maintain stable foot placement at a comfortable stance width",
    ]
    improve_bad = [
        "Prevent knees collapsing inward (valgus)",
        "Avoid heel lift—shift pressure toward heel/midfoot",
        "Reduce forward torso lean and keep a neutral spine",
        "Stabilize the pelvis (avoid lateral hip shift/asymmetry)",
        "Improve depth gradually while maintaining stability",
    ]
    return keep_good, improve_good, keep_bad, improve_bad

def llm_feedback_for_row(true_label: str, pred_label: str, confidence: float, correct: bool) -> Dict[str, str]:
    """
    Hybrid:
    - If OPENAI_API_KEY exists + USE_OPENAI_LLM=1 -> use OpenAI (gpt-4o-mini)
    - Else -> fallback to templates
    """
    if os.environ.get("USE_OPENAI_LLM", "1") == "1":
        try:
            return llm_feedback_for_row_openai(true_label, pred_label, confidence, correct)
        except Exception as e:
            print(f"⚠️ OpenAI LLM failed, falling back to templates: {e}")

    # fallback (your existing templates)
    keep_good, improve_good, keep_bad, improve_bad = _english_feedback_templates()

    if true_label == "good":
        keep = " ; ".join(random.sample(keep_good, k=min(2, len(keep_good))))
        improve = " ; ".join(random.sample(improve_good, k=min(2, len(improve_good))))
    else:
        keep = " ; ".join(random.sample(keep_bad, k=min(2, len(keep_bad))))
        improve = " ; ".join(random.sample(improve_bad, k=min(2, len(improve_bad))))

    return {"llm_keep": keep, "llm_improve": improve}


def add_llm_columns(df: pd.DataFrame) -> pd.DataFrame:
    keeps, improves = [], []
    for _, r in df.iterrows():
        fb = llm_feedback_for_row(
            true_label=str(r.get("true_label", "")),
            pred_label=str(r.get("pred_label", "")),
            confidence=float(r.get("confidence", 0.0)),
            correct=bool(r.get("correct", False)),
        )
        keeps.append(fb["llm_keep"])
        improves.append(fb["llm_improve"])

    df["llm_keep"] = keeps
    df["llm_improve"] = improves
    return df

def evaluate_split(model: nn.Module, split_dir: Path, split_name: str, transform) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates a split directory (train/val/test), writes:
      - outputs/{split_name}_predictions.csv  (ONLY requested columns)
      - outputs/confusion_{split_name}.png
      - outputs/confidence_dist_{split_name}.png
      - outputs/roc_curve_{split_name}.png

    Returns:
      df (with ONLY requested columns)
      summary dict (accuracy + f1 metrics + confusion matrix + roc_auc)
    """
    from sklearn.metrics import roc_curve, auc  # local import (or move to top)

    ds = SquatDataset(split_dir, transform=transform)
    if len(ds) == 0:
        return pd.DataFrame(), {"split": split_name, "note": "empty split"}

    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers_auto())

    model.eval()
    rows: List[Dict[str, Any]] = []
    all_y: List[int] = []
    all_p: List[int] = []
    all_score: List[float] = []  # score for class "good" (positive)

    with torch.no_grad():
        for x, y, names in loader:
            x = x.to(DEVICE)
            y_t = y.to(DEVICE) if torch.is_tensor(y) else torch.tensor(y, device=DEVICE)

            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: [B, 2]
            pred = np.argmax(probs, axis=1)

            for j in range(len(names)):
                true_y = int(y_t[j].item())
                pred_y = int(pred[j])

                score_good = float(probs[j][1])     # P(good)
                conf = float(np.max(probs[j]))      # confidence in chosen class

                rows.append({
                    "image_name": names[j],
                    "true_label": "good" if true_y == 1 else "bad",
                    "pred_label": "good" if pred_y == 1 else "bad",
                    "confidence": conf,
                    "correct": (true_y == pred_y),
                })

                all_y.append(true_y)
                all_p.append(pred_y)
                all_score.append(score_good)

    df = pd.DataFrame(rows)

    # add LLM feedback columns
    df = add_llm_columns(df)

    keep_cols = ["image_name", "true_label", "pred_label", "confidence", "correct", "llm_keep", "llm_improve"]
    df = df[keep_cols]

    # save predictions csv
    df.to_csv(OUTPUT_DIR / f"{split_name}_predictions.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(all_y, all_p, labels=[0, 1])
    save_confusion_matrix(
        cm,
        ["bad", "good"],
        OUTPUT_DIR / f"confusion_{split_name}.png",
        title=f"Confusion Matrix ({split_name})"
    )

    # confidence distribution plot
    plt.figure(figsize=(7, 5))
    corr = df[df["correct"] == True]["confidence"]
    incorr = df[df["correct"] == False]["confidence"]
    plt.hist(corr, bins=20, alpha=0.6, label="Correct")
    plt.hist(incorr, bins=20, alpha=0.6, label="Incorrect")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence: Correct vs Incorrect ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confidence_dist_{split_name}.png", dpi=150)
    plt.close()

    # ROC curve + AUC (positive class = good=1)
    y_true = np.array(all_y, dtype=np.int32)
    y_score = np.array(all_score, dtype=np.float32)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"roc_curve_{split_name}.png", dpi=150)
    plt.close()

    # classification report
    rep_txt = classification_report(
        all_y, all_p,
        labels=[0, 1],
        target_names=["bad", "good"],
        digits=4,
        zero_division=0
    )
    print(f"\n=== classification_report ({split_name}) ===\n{rep_txt}")

    rep = classification_report(
        all_y, all_p,
        labels=[0, 1],
        target_names=["bad", "good"],
        output_dict=True,
        zero_division=0
    )

    summary = {
        "split": split_name,
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean() if len(df) else 0.0),
        "confusion_matrix": cm.tolist(),
        "f1_bad": float(rep["bad"]["f1-score"]),
        "f1_good": float(rep["good"]["f1-score"]),
        "f1_macro": float(rep["macro avg"]["f1-score"]),
        "f1_weighted": float(rep["weighted avg"]["f1-score"]),
        "roc_auc": float(roc_auc),
    }

    print(
        f"\n[{split_name}] accuracy={summary['accuracy']:.4f} | "
        f"f1_macro={summary['f1_macro']:.4f} | f1_weighted={summary['f1_weighted']:.4f} | "
        f"roc_auc={summary['roc_auc']:.4f}"
    )
    return df, summary



# ============================================================
# 11) README (minimal)
# ============================================================
def write_readme(summaries: List[Dict], note: str = ""):
    lines = []
    lines.append("# PoseAITraining — Run Summary\n")
    lines.append("## Project structure (required by instructor)\n")
    lines.append("- Part 1: Synthetic data generation using 3D pose extraction + 3D pose manipulation + ControlNet Img2Img\n")
    lines.append("- Part 2: Fine-tune a pretrained ViT image model to classify Good vs Bad squats using synthetic data\n")

    lines.append("\n## Config\n")
    lines.append(f"- Device: {DEVICE}\n")
    lines.append(f"- TOTAL_IMAGES: {cfg.TOTAL_IMAGES}\n")
    lines.append(f"- VAL_RATIO: {cfg.VAL_RATIO}\n")
    lines.append(f"- TEST_RATIO: {cfg.TEST_RATIO}\n")
    lines.append(f"- WARMUP_EPOCHS: {cfg.WARMUP_EPOCHS}\n")
    lines.append(f"- FINETUNE_EPOCHS: {cfg.FINETUNE_EPOCHS}\n")
    lines.append(f"- UNFREEZE_LAST_BLOCKS: {cfg.UNFREEZE_LAST_BLOCKS}\n")
    lines.append(f"- MIXUP_ALPHA: {cfg.MIXUP_ALPHA}\n")
    lines.append(f"- LABEL_SMOOTHING: {cfg.LABEL_SMOOTHING}\n")

    lines.append("\n## Outputs\n")
    lines.append("- outputs/val_predictions.csv\n")
    lines.append("- outputs/confidence_dist_val.png\n")
    lines.append("- outputs/confusion_val.png\n")
    lines.append("- outputs/roc_curve_val.png\n")
    lines.append("- outputs/README.md\n")

    if note.strip():
        lines.append("\n## Notes\n")
        lines.append(note.strip() + "\n")

    lines.append("\n## Metrics\n")
    for s in summaries:
        lines.append(f"### {s.get('split','?')}\n")
        lines.append(f"- n = {s.get('n',0)}\n")
        lines.append(f"- accuracy = {s.get('accuracy',0):.4f}\n")
        lines.append(f"- f1_macro = {s.get('f1_macro',0):.4f}\n")
        lines.append(f"- f1_weighted = {s.get('f1_weighted',0):.4f}\n")

    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("README saved: outputs/README.md")

# ============================================================
# 12) Main
# ============================================================
def main():
    print(f"ROOT: {ROOT}")
    print(f"DEVICE: {DEVICE}")
    print(f"Config: TOTAL_IMAGES={cfg.TOTAL_IMAGES}, FINETUNE_EPOCHS={cfg.FINETUNE_EPOCHS}, MIXUP_ALPHA={cfg.MIXUP_ALPHA}")

    if not GOOD_SEEDS_DIR.exists() or not list_images(GOOD_SEEDS_DIR):
        raise FileNotFoundError("seeds/good is missing or empty. Put valid squat images into seeds/good")

    enable_gen = os.environ.get("ENABLE_GENERATION", "0") == "1"
    auto_if_empty = os.environ.get("AUTO_GENERATE_IF_EMPTY", "1") == "1"

    n_train = count_images(TRAIN_DIR/"good") + count_images(TRAIN_DIR/"bad")
    n_val   = count_images(VAL_DIR/"good") + count_images(VAL_DIR/"bad")

    if enable_gen or (auto_if_empty and (n_train == 0 or n_val == 0)):
        generate_synthetic_dataset(cfg.TOTAL_IMAGES)
    else:
        print("Generation skipped (ENABLE_GENERATION!=1 and dataset not empty). Using existing synthetic_dataset/")

    n_train = count_images(TRAIN_DIR/"good") + count_images(TRAIN_DIR/"bad")
    n_val   = count_images(VAL_DIR/"good") + count_images(VAL_DIR/"bad")
    if n_train == 0 or n_val == 0:
        print(" Dataset is still empty. Check paths: synthetic_dataset/train/{good,bad} and synthetic_dataset/val/{good,bad}")
        print(" Tip: set ENABLE_GENERATION=1 to force generation (CUDA recommended).")
        return

    model, transform, train_info = train_vit()
    if model is None or transform is None:
        print("Training failed.")
        return


    summaries: List[Dict] = []
    _, val_summary = evaluate_split(model, VAL_DIR, "val", transform)
    summaries.append(val_summary)

    print("\n=== Final Accuracy ===")
    for s in summaries:
        print(f"{s['split']}: accuracy={s['accuracy']:.4f} (n={s.get('n', 0)})")

    note = "Generation uses 3D pose extraction + 3D pose manipulation (dragging + rotations) + ControlNet Img2Img. Training uses ViT fine-tuning with cosine LR, label smoothing, optional mixup, early stopping."


    write_readme(summaries, note=note)

    for s in summaries:
        print(f"{s['split']}: acc={s['accuracy']:.4f} | f1_macro={s.get('f1_macro',0):.4f} | f1_weighted={s.get('f1_weighted',0):.4f}")

    if MINIMAL_OUTPUT:
        prune_outputs_dir()

    print("\n✅ DONE. Check outputs/")

if __name__ == "__main__":
    main()
