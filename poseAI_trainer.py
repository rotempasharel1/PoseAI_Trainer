from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import shutil
import zipfile
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------
# 0) Dependency setup (Colab-friendly, with REAL validation)
# -----------------------
def _in_colab() -> bool:
    try:
        import google.colab  
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
    except Exception as e:
        print("Installing/repairing dependencies...")
        print("Reason:", repr(e))
        _pip_install(needed)
        import importlib
        importlib.invalidate_caches()


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from sklearn.metrics import confusion_matrix, classification_report

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from diffusers import ControlNetModel
try:
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
except Exception:
    from diffusers.pipelines.controlnet import StableDiffusionControlNetImg2ImgPipeline  # type: ignore
from diffusers import DPMSolverMultistepScheduler


# -----------------------
# 1) Config
# -----------------------
@dataclass
class CFG:
    TOTAL_IMAGES: int = 2000

    VAL_RATIO: float = 0.2
    TEST_RATIO: float = 0.0 
    IMG_SIZE: int = 512
    CONTROL_SIZE: int = 512
    CANVAS_MARGIN: float = 0.08

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

    IMG2IMG_STRENGTH_GOOD: float = 0.75
    IMG2IMG_STRENGTH_BAD: float = 0.80

    NUM_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    CONTROLNET_SCALE: float = 1.0

    SYNTH_BAD_FROM_BADSEEDS: bool = True
    BAD_FROM_BADSEEDS_FRACTION_TRAIN: float = 0.25

    WARMUP_EPOCHS: int = 1
    FINETUNE_EPOCHS: int = 4
    UNFREEZE_LAST_BLOCKS: int = 4

    BATCH_SIZE: int = 16
    LR_WARMUP: float = 2e-4
    LR_FINETUNE: float = 6e-5
    WEIGHT_DECAY: float = 1e-4

    SEED: int = 42

    ZIP_NAME: str = "PoseAITraining_artifact.zip"

cfg = CFG()


# -----------------------
# 2) Paths
# -----------------------
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


# -----------------------
# 3) Utilities
# -----------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

seed_everything(cfg.SEED)

def list_images(p: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    out: List[Path] = []
    for e in exts:
        out += list(p.glob(e))
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


# -----------------------
# 4) MediaPipe 3D pose extraction
# -----------------------
mp_pose = mp.solutions.pose

def extract_world_landmarks_from_image(pil_img: Image.Image) -> Optional[landmark_pb2.LandmarkList]:
    img = np.array(pil_img)  # RGB
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        res = pose.process(img)
        return res.pose_world_landmarks

def _lm_vec(lms, idx: int) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


# -----------------------
# 5) Tree skeleton + FK
# -----------------------
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

def build_skeleton_tree(world_landmarks) -> Tuple[SkeletonNode, Dict[str, SkeletonNode]]:
    lms = world_landmarks.landmark

    LH = mp_pose.PoseLandmark.LEFT_HIP.value
    RH = mp_pose.PoseLandmark.RIGHT_HIP.value
    LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    mid_hip = (_lm_vec(lms, LH) + _lm_vec(lms, RH)) / 2.0
    mid_shoulder = (_lm_vec(lms, LS) + _lm_vec(lms, RS)) / 2.0

    def get_centered(idx: int) -> np.ndarray:
        return _lm_vec(lms, idx) - mid_hip

    root = SkeletonNode("MID_HIP", None, None)
    root.world_pos = np.zeros(3, dtype=np.float32)

    spine = SkeletonNode("MID_SHOULDER", root, None)
    spine.world_pos = (mid_shoulder - mid_hip).astype(np.float32)

    node_map: Dict[str, SkeletonNode] = {"MID_HIP": root, "MID_SHOULDER": spine}

    l_hip = SkeletonNode("LEFT_HIP", root, LH);  l_hip.world_pos = get_centered(LH)
    l_knee = SkeletonNode("LEFT_KNEE", l_hip, mp_pose.PoseLandmark.LEFT_KNEE.value); l_knee.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_KNEE.value)
    l_ankle = SkeletonNode("LEFT_ANKLE", l_knee, mp_pose.PoseLandmark.LEFT_ANKLE.value); l_ankle.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_ANKLE.value)

    r_hip = SkeletonNode("RIGHT_HIP", root, RH); r_hip.world_pos = get_centered(RH)
    r_knee = SkeletonNode("RIGHT_KNEE", r_hip, mp_pose.PoseLandmark.RIGHT_KNEE.value); r_knee.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle = SkeletonNode("RIGHT_ANKLE", r_knee, mp_pose.PoseLandmark.RIGHT_ANKLE.value); r_ankle.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

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

    return root, node_map

def clone_skeleton(root: SkeletonNode) -> Tuple[SkeletonNode, Dict[str, SkeletonNode]]:
    node_map: Dict[str, SkeletonNode] = {}
    def _clone(node: SkeletonNode, parent: Optional[SkeletonNode]) -> SkeletonNode:
        nn = SkeletonNode(node.name, parent, node.landmark_idx)
        nn.local_pos = node.local_pos.copy()
        nn.world_pos = node.world_pos.copy()
        node_map[nn.name] = nn
        for ch in node.children:
            _clone(ch, nn)
        return nn
    new_root = _clone(root, None)
    return new_root, node_map

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

def make_bad_pose_fk(root: SkeletonNode, node_map: Dict[str, SkeletonNode]):
    # FK-only realistic bad variety
    def rng(a, b): return random.uniform(a, b)

    faults = []
    if random.random() < 0.90: faults.append("forward_lean")
    if random.random() < 0.70: faults.append("knee_valgus")
    if random.random() < 0.45: faults.append("knee_forward")
    if random.random() < 0.40: faults.append("hip_shift")
    if random.random() < 0.30: faults.append("torso_twist")
    if random.random() < 0.30: faults.append("pelvic_tuck")

    if "forward_lean" in faults and "MID_SHOULDER" in node_map:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(25, 50)))
        if random.random() < 0.35:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(rng(-10, 10)))

    if "torso_twist" in faults and "MID_SHOULDER" in node_map:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_y(rng(-20, 20)))
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(rng(-12, 12)))

    if "knee_valgus" in faults:
        if "LEFT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_KNEE"], rot_z(rng(+12, +28)))
        if "RIGHT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_KNEE"], rot_z(rng(-12, -28)))
        if random.random() < 0.40:
            if "LEFT_HIP" in node_map:
                apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_y(rng(+6, +16)))
            if "RIGHT_HIP" in node_map:
                apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_y(rng(-6, -16)))

    if "knee_forward" in faults:
        sgn = 1.0 if random.random() < 0.5 else -1.0
        if "LEFT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_KNEE"], rot_x(sgn * rng(10, 25)))
        if "RIGHT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_KNEE"], rot_x(sgn * rng(10, 25)))
        if "MID_SHOULDER" in node_map and random.random() < 0.6:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(8, 18)))

    if "hip_shift" in faults:
        shift_dir = 1.0 if random.random() < 0.5 else -1.0
        if "LEFT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_z(shift_dir * rng(6, 16)))
        if "RIGHT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_z(-shift_dir * rng(6, 16)))
        if "MID_SHOULDER" in node_map and random.random() < 0.7:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(-shift_dir * rng(4, 10)))

    if "pelvic_tuck" in faults:
        if "LEFT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_x(rng(8, 22)))
        if "RIGHT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_x(rng(8, 22)))
        if "MID_SHOULDER" in node_map and random.random() < 0.8:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(6, 16)))

    update_world_positions(root)


# -----------------------
# 6) Camera sampling + projection + OpenPose-like map
# -----------------------
def sample_camera_params() -> Dict[str, float]:
    return {
        "yaw": random.uniform(-45, 45),
        "pitch": random.uniform(-12, 12),
        "roll": random.uniform(-6, 6),
        "distance": random.uniform(3.0, 5.0),
        "fov": random.uniform(30, 50),
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
        cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
    return Image.fromarray(canvas)


# -----------------------
# 7) Stage 1: SD Img2Img + ControlNet generation
# -----------------------
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
    print(f"\n=== Stage 1: Synthetic dataset (SD Img2Img + ControlNet) on {DEVICE} ===")

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

    test_good_seeds: List[Path] = []
    if cfg.TEST_RATIO > 0 and len(train_good_seeds) >= 3:
        n_test = max(1, int(round(n_good * cfg.TEST_RATIO)))
        test_good_seeds = train_good_seeds[:n_test]
        train_good_seeds = train_good_seeds[n_test:]

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

    pose_cache: Dict[str, Tuple[SkeletonNode, Dict[str, SkeletonNode], Image.Image]] = {}

    def get_pose_and_init(seed_path: Path) -> Optional[Tuple[SkeletonNode, Dict[str, SkeletonNode], Image.Image]]:
        key = str(seed_path.resolve())
        if key in pose_cache:
            return pose_cache[key]
        init_img = pil_center_crop_resize(safe_imread_pil(seed_path), cfg.IMG_SIZE)
        wlms = extract_world_landmarks_from_image(init_img)
        if wlms is None:
            return None
        root, node_map = build_skeleton_tree(wlms)
        pose_cache[key] = (root, node_map, init_img)
        return pose_cache[key]

    def gen_one(split: str, label: str, seed_path: Path, source: str):
        pack = get_pose_and_init(seed_path)
        if pack is None:
            return False

        root_rest, _node_map_rest, init_img = pack
        cam = sample_camera_params()

        root, node_map = clone_skeleton(root_rest)
        if label == "bad" and source == "bad_from_good":
            make_bad_pose_fk(root, node_map)

        pts3d = skeleton_to_landmark_dict(root)
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

        with torch.autocast(device_type="cuda", dtype=TORCH_DTYPE) if DEVICE == "cuda" else torch.no_grad():
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

        fname = f"synth_{label}_{source}_{seed_path.stem}_{now_id()}.jpg"
        out_path = SYNTH_DIR / split / label / fname
        _save_image(out, out_path)
        return True

    def rr(seeds: List[Path]):
        i = 0
        while True:
            yield seeds[i % len(seeds)] if seeds else None
            i += 1

    rr_train_good = rr(train_good_seeds)
    rr_val_good   = rr(val_good_seeds if val_good_seeds else train_good_seeds)
    rr_test_good  = rr(test_good_seeds if test_good_seeds else (val_good_seeds if val_good_seeds else train_good_seeds))
    rr_bad        = rr(bad_seeds)

    train_bad_target = targets["train"]["bad"]
    train_bad_from_badseeds = 0
    if cfg.SYNTH_BAD_FROM_BADSEEDS and bad_seeds and train_bad_target > 0:
        train_bad_from_badseeds = int(round(train_bad_target * cfg.BAD_FROM_BADSEEDS_FRACTION_TRAIN))
        train_bad_from_badseeds = max(0, min(train_bad_from_badseeds, train_bad_target))
    train_bad_from_good = train_bad_target - train_bad_from_badseeds

    def fill_split(split: str, good_target: int, bad_from_good_target: int):
        while existing[split]["good"] < good_target:
            seed_path = next(rr_train_good if split == "train" else rr_val_good if split == "val" else rr_test_good)
            if seed_path is None:
                break
            ok = gen_one(split, "good", seed_path, source="good_from_goodseeds")
            if ok:
                existing[split]["good"] += 1

        while existing[split]["bad"] < bad_from_good_target:
            seed_path = next(rr_train_good if split == "train" else rr_val_good if split == "val" else rr_test_good)
            if seed_path is None:
                break
            ok = gen_one(split, "bad", seed_path, source="bad_from_good")
            if ok:
                existing[split]["bad"] += 1

    fill_split("train", targets["train"]["good"], train_bad_from_good)

    while existing["train"]["bad"] < targets["train"]["bad"] and train_bad_from_badseeds > 0:
        seed_path = next(rr_bad)
        if seed_path is None:
            break
        ok = gen_one("train", "bad", seed_path, source="extra_bad_from_badseeds")
        if ok:
            existing["train"]["bad"] += 1
            train_bad_from_badseeds -= 1

    fill_split("val", targets["val"]["good"], targets["val"]["bad"])
    fill_split("test", targets["test"]["good"], targets["test"]["bad"])

    print("\n✅ Generation complete. Final counts:", {
        sp: {"good": count_images(SYNTH_DIR/sp/"good"), "bad": count_images(SYNTH_DIR/sp/"bad")}
        for sp in ["train", "val", "test"]
    })


# -----------------------
# 8) EDA (expanded) — NO sample montages
# -----------------------
def _parse_source_from_name(name: str) -> str:
    m = re.match(r"synth_(good|bad)_([A-Za-z0-9_]+)_", name)
    return m.group(2) if m else "unknown"

def _image_stats_one(path: Path) -> Dict[str, float]:
    img = safe_imread_pil(path)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size_kb = float(path.stat().st_size / 1024.0)
    return {"brightness": brightness, "contrast": contrast, "sharpness": sharpness, "size_kb": size_kb}

def run_eda(max_per_bucket: int = 250):
    print("\n=== EDA (expanded) ===")

    counts = []
    for split in ["train", "val", "test"]:
        for label in ["good", "bad"]:
            paths = list_images(SYNTH_DIR / split / label)
            counts.append({"split": split, "label": label, "count": len(paths)})
    counts_df = pd.DataFrame(counts)

    pivot = counts_df.pivot(index="split", columns="label", values="count").fillna(0)
    ax = pivot.plot(kind="bar")
    ax.set_title("Class counts per split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_class_counts.png", dpi=150)
    plt.close(fig)

    counts_df.to_csv(OUTPUT_DIR / "eda_counts.csv", index=False)

    src_rows = []
    for split in ["train", "val", "test"]:
        for label in ["good", "bad"]:
            for p in list_images(SYNTH_DIR / split / label):
                src_rows.append({"split": split, "label": label, "source": _parse_source_from_name(p.name)})
    src_df = pd.DataFrame(src_rows) if src_rows else pd.DataFrame(columns=["split","label","source"])
    if not src_df.empty:
        src_pivot = src_df.groupby(["split","label","source"]).size().reset_index(name="count")
        src_pivot.to_json(OUTPUT_DIR / "eda_source_counts.json", orient="records", indent=2)
        src_pivot.to_csv(OUTPUT_DIR / "eda_source_counts.csv", index=False)

    stats_rows = []
    for split in ["train", "val", "test"]:
        for label in ["good", "bad"]:
            paths = list_images(SYNTH_DIR / split / label)
            if not paths:
                continue
            take = paths[:max_per_bucket]
            for p in take:
                st = _image_stats_one(p)
                st.update({"split": split, "label": label})
                stats_rows.append(st)

    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()
    if not stats_df.empty:
        for col in ["brightness", "sharpness", "size_kb"]:
            fig = plt.figure()
            for (split, label), g in stats_df.groupby(["split","label"]):
                plt.hist(g[col].values, bins=30, alpha=0.5, label=f"{split}-{label}")
            plt.title(f"EDA histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.legend()
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"eda_hist_{col}.png", dpi=150)
            plt.close(fig)

        summary = (
            stats_df.groupby(["split","label"])[["brightness","contrast","sharpness","size_kb"]]
            .agg(["mean","std","min","max"])
            .reset_index()
        )
        summary_records = []
        for _, row in summary.iterrows():
            rec = {"split": row[("split","")], "label": row[("label","")]}
            for metric in ["brightness","contrast","sharpness","size_kb"]:
                rec[f"{metric}_mean"] = float(row[(metric,"mean")])
                rec[f"{metric}_std"]  = float(row[(metric,"std")])
                rec[f"{metric}_min"]  = float(row[(metric,"min")])
                rec[f"{metric}_max"]  = float(row[(metric,"max")])
            summary_records.append(rec)
        (OUTPUT_DIR / "eda_stats_summary.json").write_text(json.dumps(summary_records, indent=2), encoding="utf-8")

        stats_df.to_csv(OUTPUT_DIR / "eda_stats.csv", index=False)

    report = {
        "device": DEVICE,
        "counts": counts,
        "notes": "No sample montages. EDA data saved as CSV and JSON.",
    }
    (OUTPUT_DIR / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✅ EDA saved: outputs/eda_*.png + outputs/eda_*json + outputs/eda_*.csv")


# -----------------------
# 9) Stage 2: ViT training (improved)
# -----------------------
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
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
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

def train_vit() -> Tuple[Optional[nn.Module], Optional[transforms.Compose]]:
    print("\n=== Stage 2: Train image model (ViT-B/16) — improved ===")

    train_tf, eval_tf = _make_transforms()

    train_ds = SquatDataset(TRAIN_DIR, transform=train_tf)
    val_ds = SquatDataset(VAL_DIR, transform=eval_tf)

    if len(train_ds) == 0:
        print("No training data found in synthetic_dataset/train.")
        return None, None

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    num_f = model.heads.head.in_features
    model.heads.head = nn.Linear(num_f, 2)
    model.to(DEVICE)

    _unfreeze_last_blocks(model, n_blocks=0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg.LR_WARMUP, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_acc = -1.0
    best_path = OUTPUT_DIR / "vit_squat_best.pth"

    def run_epoch(train: bool) -> float:
        loader = train_loader if train else val_loader
        model.train(train)
        accs = []
        losses = []
        for x, y, _ in loader:
            x = x.to(DEVICE)
            y = torch.tensor(y, device=DEVICE) if not torch.is_tensor(y) else y.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.append(float(loss.item()))
            accs.append(_accuracy_from_logits(logits.detach(), y))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_acc  = float(np.mean(accs)) if accs else 0.0
        return mean_loss, mean_acc

    for epoch in range(cfg.WARMUP_EPOCHS):
        tr_loss, tr_acc = run_epoch(train=True)
        va_loss, va_acc = run_epoch(train=False)
        print(f"[Warmup {epoch+1}/{cfg.WARMUP_EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    _unfreeze_last_blocks(model, n_blocks=cfg.UNFREEZE_LAST_BLOCKS)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg.LR_FINETUNE, weight_decay=cfg.WEIGHT_DECAY)

    for epoch in range(cfg.FINETUNE_EPOCHS):
        tr_loss, tr_acc = run_epoch(train=True)
        va_loss, va_acc = run_epoch(train=False)
        print(f"[Finetune {epoch+1}/{cfg.FINETUNE_EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    torch.save(model.state_dict(), OUTPUT_DIR / "vit_squat.pth")
    print(f"✅ Saved model to outputs/vit_squat.pth (best val acc={best_val_acc:.4f})")
    return model, eval_tf


# -----------------------
# 10) Evaluation + predictions CSV (with LLM columns) + confusion matrix
# -----------------------
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

def _fallback_feedback(true_label: str, pred_label: str, confidence: float) -> Dict[str, str]:
    keep_1 = "Keep full-body framing with hips/knees/ankles visible."
    keep_2 = "Keep consistent lighting and avoid motion blur."

    if true_label == "good" and pred_label == "bad":
        improve_1 = "Add more GOOD variety (angles/body types) and keep knees tracking over toes + neutral spine."
        improve_2 = "Reduce forward-lean cues in GOOD images; ensure heels stay flat."
    elif true_label == "bad" and pred_label == "good":
        improve_1 = "Add more BAD examples emphasizing knee valgus + heel lift + excessive torso lean."
        improve_2 = "Add 'borderline' bad squats (subtle faults) to sharpen the decision boundary."
    else:
        improve_1 = "Add more viewpoint diversity and backgrounds to improve robustness."
        improve_2 = "Increase pose-fault diversity (valgus, hip shift, butt wink, heel lift)."

    summary = f"pred={pred_label} conf={confidence:.2f}. keep: {keep_1} | improve: {improve_1}"
    return {
        "llm_generator": "fallback",
        "keep_1": keep_1,
        "keep_2": keep_2,
        "improve_1": improve_1,
        "improve_2": improve_2,
        "llm_summary": summary,
    }

def _openai_feedback_optional(true_label: str, pred_label: str, confidence: float) -> Optional[Dict[str, str]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import openai  
    except Exception:
        return None

    try:
        msg = (
            "Return JSON with keys: keep_1, keep_2, improve_1, improve_2, llm_summary.\n"
            f"True label: {true_label}\nPred label: {pred_label}\nConfidence: {confidence:.2f}\n"
            "Style: short coaching tips."
        )
        client = openai.OpenAI(api_key=api_key) 
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content": msg}],
            temperature=0.2
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        return {
            "llm_generator": "openai",
            "keep_1": str(data.get("keep_1","")).strip(),
            "keep_2": str(data.get("keep_2","")).strip(),
            "improve_1": str(data.get("improve_1","")).strip(),
            "improve_2": str(data.get("improve_2","")).strip(),
            "llm_summary": str(data.get("llm_summary","")).strip(),
        }
    except Exception:
        return None

def _attach_llm_columns(df: pd.DataFrame, max_openai: int = 60) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()
    for col in ["llm_generator","keep_1","keep_2","improve_1","improve_2","llm_summary"]:
        df[col] = ""

    cand = df.copy()
    cand["rank"] = 0
    cand.loc[cand["correct"] == False, "rank"] = 2
    cand.loc[(cand["correct"] == True) & (cand["confidence"] < 0.60), "rank"] = 1
    cand = cand.sort_values(["rank","confidence"], ascending=[False, True]).head(max_openai)

    openai_idx = set(cand.index.tolist())

    for i, row in df.iterrows():
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        conf = float(row["confidence"])

        fb = None
        if i in openai_idx:
            fb = _openai_feedback_optional(true_label, pred_label, conf)

        if fb is None:
            fb = _fallback_feedback(true_label, pred_label, conf)

        for k, v in fb.items():
            df.at[i, k] = v

    return df

def evaluate_split(model: nn.Module, split_dir: Path, split_name: str, transform) -> Tuple[pd.DataFrame, Dict]:
    ds = SquatDataset(split_dir, transform=transform)
    if len(ds) == 0:
        return pd.DataFrame(), {"split": split_name, "note": "empty split"}

    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    rows = []
    all_y = []
    all_p = []

    with torch.no_grad():
        for x, y, names in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)

            for j in range(len(names)):
                true_y = int(y[j])
                pred_y = int(pred[j])
                conf = float(np.max(probs[j]))
                rows.append({
                    "split": split_name,
                    "image_name": names[j],
                    "true_label": "good" if true_y == 1 else "bad",
                    "pred_label": "good" if pred_y == 1 else "bad",
                    "confidence": conf,
                    "correct": (true_y == pred_y),
                    "p_bad": float(probs[j][0]),
                    "p_good": float(probs[j][1]),
                })
                all_y.append(true_y)
                all_p.append(pred_y)

    df = pd.DataFrame(rows)

    df = _attach_llm_columns(df)

    df.to_csv(OUTPUT_DIR / f"{split_name}_predictions.csv", index=False)

    cm = confusion_matrix(all_y, all_p, labels=[0, 1])
    save_confusion_matrix(cm, ["bad", "good"], OUTPUT_DIR / f"confusion_{split_name}.png",
                          title=f"Confusion Matrix ({split_name})")

    rep = classification_report(all_y, all_p, target_names=["bad", "good"], output_dict=True)
    with open(OUTPUT_DIR / f"classification_report_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    summary = {
        "split": split_name,
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean() if len(df) else 0.0),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n[{split_name}] accuracy={summary['accuracy']:.4f}")
    return df, summary


# -----------------------
# 11) README
# -----------------------
def write_readme(summaries: List[Dict], note: str = ""):
    lines = []
    lines.append("# PoseAITraining — Run Summary\n")
    lines.append("## Config\n")
    lines.append(f"- Device: {DEVICE}\n")
    lines.append(f"- TOTAL_IMAGES: {cfg.TOTAL_IMAGES}\n")
    lines.append(f"- VAL_RATIO: {cfg.VAL_RATIO}\n")
    lines.append(f"- TEST_RATIO: {cfg.TEST_RATIO}\n")
    lines.append(f"- WARMUP_EPOCHS: {cfg.WARMUP_EPOCHS}\n")
    lines.append(f"- FINETUNE_EPOCHS: {cfg.FINETUNE_EPOCHS}\n")
    lines.append(f"- UNFREEZE_LAST_BLOCKS: {cfg.UNFREEZE_LAST_BLOCKS}\n")
    lines.append("\n## Outputs\n")
    lines.append("- `outputs/eda_*.png` + `outputs/eda_*json` + `outputs/eda_*.csv` (expanded EDA)\n")
    lines.append("- `outputs/*_predictions.csv` (includes keep/improve/summary columns)\n")
    lines.append("- `outputs/confusion_*.png` + `outputs/classification_report_*.json`\n")
    lines.append("- `outputs/vit_squat.pth` + `outputs/vit_squat_best.pth`\n")
    lines.append("- `outputs/PoseAITraining_artifact.zip`\n")

    if note.strip():
        lines.append("\n## Notes\n")
        lines.append(note.strip() + "\n")

    lines.append("\n## Metrics\n")
    for s in summaries:
        lines.append(f"### {s.get('split','?')}\n")
        lines.append(f"- n = {s.get('n',0)}\n")
        lines.append(f"- accuracy = {s.get('accuracy',0):.4f}\n")

    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("README saved: outputs/README.md")


# -----------------------
# 12) Zip artifact + download
# -----------------------
def zip_artifact():
    zip_path = OUTPUT_DIR / cfg.ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if SYNTH_DIR.exists():
            for p in SYNTH_DIR.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(ROOT)))
        if OUTPUT_DIR.exists():
            for p in OUTPUT_DIR.rglob("*"):
                if p.is_file() and p.name != cfg.ZIP_NAME:
                    z.write(p, arcname=str(p.relative_to(ROOT)))
        # include main.py
        try:
            this_file = Path(__file__).resolve()
            if this_file.exists():
                z.write(this_file, arcname=str(this_file.relative_to(ROOT)))
        except Exception:
            pass

    print(f"zip created: {zip_path}")

    if _in_colab():
        try:
            from google.colab import files 
            files.download(str(zip_path))
        except Exception:
            pass


# -----------------------
# 13) Main
# -----------------------
def main():
    print(f"ROOT: {ROOT}")
    print(f"DEVICE: {DEVICE}")

    if not GOOD_SEEDS_DIR.exists():
        raise FileNotFoundError("❌ seeds/good folder missing.")
    if not list_images(GOOD_SEEDS_DIR):
        print("⚠️ No seed images found in seeds/good. Proceeding with existing synthetic_dataset/ (generation disabled).")
        # return  # commented to allow proceeding

    # ============================
    # Stage 1 — GENERATION (DISABLED BY REQUEST)
    # ============================
    # אם תרצי להפעיל שוב:
    #   - להסיר את ה-# בשורה למטה
    #   - או להגדיר: os.environ["ENABLE_GENERATION"]="1"
    #
    # if os.environ.get("ENABLE_GENERATION","0") == "1":
    #     generate_synthetic_dataset(cfg.TOTAL_IMAGES)
    # else:
    #     print("⚠️ Generation disabled (ENABLE_GENERATION=0). Using existing synthetic_dataset/")

    print(" Generation disabled (by request). Using existing synthetic_dataset/")

    n_train = count_images(TRAIN_DIR/"good") + count_images(TRAIN_DIR/"bad")
    n_val   = count_images(VAL_DIR/"good")   + count_images(VAL_DIR/"bad")
    if n_train == 0 or n_val == 0:
        print("❌ No dataset found. You disabled generation, but synthetic_dataset/ is empty.")
        print("👉 Set ENABLE_GENERATION=1 and rerun if you want to generate.")
        return

    if os.environ.get("SKIP_EDA", "0") != "1":
        run_eda()
    else:
        print("⚠️ SKIP_EDA=1 -> skipping EDA")

    model, transform = (None, None)
    if os.environ.get("SKIP_TRAIN", "0") != "1":
        model, transform = train_vit()
    else:
        print("⚠️ SKIP_TRAIN=1 -> skipping training, trying to load existing model")

    if model is None or transform is None:
        best_path = OUTPUT_DIR / "vit_squat_best.pth"
        if best_path.exists():
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            num_f = model.heads.head.in_features
            model.heads.head = nn.Linear(num_f, 2)
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
            model.to(DEVICE)
            _, transform = _make_transforms()
            print("✅ Loaded existing model from vit_squat_best.pth")
        else:
            print("❌ No model found, cannot evaluate.")
            return

    val_df, val_summary = evaluate_split(model, VAL_DIR, "val", transform)

    test_note = ""
    if count_images(TEST_DIR/"good") + count_images(TEST_DIR/"bad") == 0:
        test_note = "TEST split is empty (TEST_RATIO=0). For convenience, test reports reuse val."
        test_df = val_df.copy()
        test_df["split"] = "test"
        test_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
        shutil.copyfile(OUTPUT_DIR / "confusion_val.png", OUTPUT_DIR / "confusion_test.png")
        shutil.copyfile(OUTPUT_DIR / "classification_report_val.json", OUTPUT_DIR / "classification_report_test.json")
        test_summary = {
            "split": "test",
            "n": int(len(test_df)),
            "accuracy": float(test_df["correct"].mean() if len(test_df) else 0.0),
            "confusion_matrix": val_summary.get("confusion_matrix", []),
        }
    else:
        _test_df, test_summary = evaluate_split(model, TEST_DIR, "test", transform)

    write_readme([val_summary, test_summary], note=test_note)
    zip_artifact()

    print("\n✅ DONE. Check outputs/")

if __name__ == "__main__":
    main()
