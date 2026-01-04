# main.py — PoseAITraining (Diagram-Accurate 1:1)
# Paste this entire file into your Colab project as main.py and run:
#   %cd /content/project/PoseAITraining
#   !python main.py
#
# Required folders:
#   seeds/good  (at least 5-10 seed images)
#   seeds/bad   (optional; used for "Extra: real BAD seeds (train only)")
#
# Output:
#   synthetic_dataset/{train,val,test}/{good,bad}/  (generated images)
#   outputs/  (EDA, confusion matrices, CSV tables, feedback JSON, README, zip artifact)

import os
import re
import io
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
# 0) Robust dependency setup (Colab-friendly)
# -----------------------
def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def _pip_install(pkgs: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.check_call(cmd)

def ensure_deps():
    # NOTE: If this installs packages, you can re-run the cell/command after it finishes.
    needed = [
        "mediapipe==0.10.14",
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
    # Torch/torchvision usually exist in Colab GPU images. We don't force-install them here.
    try:
        import mediapipe  # noqa
        import diffusers  # noqa
        import transformers  # noqa
        import accelerate  # noqa
        import cv2  # noqa
        import PIL  # noqa
        import numpy  # noqa
        import pandas  # noqa
        import sklearn  # noqa
        import matplotlib  # noqa
        import tqdm  # noqa
    except Exception:
        print("📦 Installing missing dependencies...")
        _pip_install(needed)
        # invalidate caches
        import importlib
        importlib.invalidate_caches()

ensure_deps()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

import mediapipe as mp
from sklearn.metrics import confusion_matrix, classification_report

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# diffusers imports (img2img + ControlNet)
from diffusers import ControlNetModel
try:
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
except Exception:
    # Older naming fallback (rare)
    from diffusers.pipelines.controlnet import StableDiffusionControlNetImg2ImgPipeline  # type: ignore
from diffusers import DPMSolverMultistepScheduler


# -----------------------
# 1) Config (matches diagram, with safe defaults)
# -----------------------
@dataclass
class CFG:
    # Dataset
    TOTAL_IMAGES: int = 2000

    # Diagram decision: split GOOD seeds to train/val
    VAL_RATIO: float = 0.2
    # Diagram doesn't mention test; we support it without breaking diagram:
    # if TEST_RATIO == 0, "test" evaluation will reuse val (and CSV/CM still exported).
    TEST_RATIO: float = 0.0

    # Generation resolution (diagram says normalize to 512x512)
    IMG_SIZE: int = 512
    CONTROL_SIZE: int = 512
    CANVAS_MARGIN: float = 0.08  # "canvas (margin)" in diagram

    # Stable Diffusion (Img2Img) + ControlNet OpenPose
    SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_MODEL: str = "lllyasviel/sd-controlnet-openpose"

    # Prompts
    PROMPT_GOOD: str = (
        "Realistic photo of a person performing a perfect bodyweight squat in a modern gym, "
        "heels down, knees tracking over toes, neutral spine, sharp focus, natural lighting."
    )
    PROMPT_BAD: str = (
        "Realistic photo of a person performing a squat with bad form in a gym, "
        "knees collapsing inward, torso leaning too far forward, unstable posture, sharp focus."
    )
    NEGATIVE_PROMPT: str = (
        "cartoon, illustration, anime, low quality, blurry, deformed, extra limbs, extra fingers, "
        "bad anatomy, bad proportions, distorted face, watermark, text, logo"
    )

    # Img2Img strength
    IMG2IMG_STRENGTH_GOOD: float = 0.75
    IMG2IMG_STRENGTH_BAD: float = 0.80

    # Diffusion steps
    NUM_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    CONTROLNET_SCALE: float = 1.0

    # Extra branch (diagram): real BAD seeds (train only)
    SYNTH_BAD_FROM_BADSEEDS: bool = True
    # How many of the "bad" images (train split) we try to source from bad seeds (within TOTAL_IMAGES)
    BAD_FROM_BADSEEDS_FRACTION_TRAIN: float = 0.25

    # Stage 2 (ViT)
    FREEZE_VIT_BACKBONE: bool = True
    EPOCHS: int = 3
    BATCH_SIZE: int = 8
    LR: float = 1e-4

    # Repro
    SEED: int = 42

    # Output artifact zip
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
    out = []
    for e in exts:
        out += list(p.glob(e))
    return sorted(out)

def safe_imread_pil(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

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
# 4) MediaPipe 3D pose extraction (diagram: Extract 3D pose with MediaPipe per image)
# -----------------------
mp_pose = mp.solutions.pose

def extract_world_landmarks_from_image(pil_img: Image.Image) -> Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]:
    # Use MediaPipe on RGB image
    img = np.array(pil_img)[:, :, ::-1]  # RGB->BGR for cv2? Actually MP expects RGB, so keep RGB:
    img = np.array(pil_img)  # RGB
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        res = pose.process(img)
        return res.pose_world_landmarks

def _lm_vec(lms, idx: int) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


# -----------------------
# 5) Tree skeleton + FK (diagram: Build tree-structured skeleton + rest offsets, apply joint rotations keep bone lengths)
# -----------------------
class SkeletonNode:
    def __init__(self, name: str, parent: Optional["SkeletonNode"]=None, landmark_idx: Optional[int]=None):
        self.name = name
        self.parent = parent
        self.children: List["SkeletonNode"] = []
        self.landmark_idx = landmark_idx  # MediaPipe landmark index
        self.local_pos = np.zeros(3, dtype=np.float32)  # offset from parent (bone vector)
        self.world_pos = np.zeros(3, dtype=np.float32)
        if parent is not None:
            parent.children.append(self)

def build_skeleton_tree(world_landmarks) -> Tuple[SkeletonNode, Dict[str, SkeletonNode]]:
    """
    Build a tree skeleton from MediaPipe world landmarks.
    We center everything at MID_HIP so root is at origin.
    """
    lms = world_landmarks.landmark

    # MediaPipe indices
    LH = mp_pose.PoseLandmark.LEFT_HIP.value
    RH = mp_pose.PoseLandmark.RIGHT_HIP.value
    LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    mid_hip = (_lm_vec(lms, LH) + _lm_vec(lms, RH)) / 2.0
    mid_shoulder = (_lm_vec(lms, LS) + _lm_vec(lms, RS)) / 2.0

    def get_centered(idx: int) -> np.ndarray:
        return _lm_vec(lms, idx) - mid_hip

    # Root (virtual)
    root = SkeletonNode("MID_HIP", parent=None, landmark_idx=None)
    root.world_pos = np.zeros(3, dtype=np.float32)

    # Spine (virtual)
    spine = SkeletonNode("MID_SHOULDER", parent=root, landmark_idx=None)
    spine.world_pos = (mid_shoulder - mid_hip).astype(np.float32)

    node_map = {"MID_HIP": root, "MID_SHOULDER": spine}

    # Build legs
    l_hip = SkeletonNode("LEFT_HIP", parent=root, landmark_idx=LH);  l_hip.world_pos = get_centered(LH)
    l_knee = SkeletonNode("LEFT_KNEE", parent=l_hip, landmark_idx=mp_pose.PoseLandmark.LEFT_KNEE.value); l_knee.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_KNEE.value)
    l_ankle = SkeletonNode("LEFT_ANKLE", parent=l_knee, landmark_idx=mp_pose.PoseLandmark.LEFT_ANKLE.value); l_ankle.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_ANKLE.value)

    r_hip = SkeletonNode("RIGHT_HIP", parent=root, landmark_idx=RH); r_hip.world_pos = get_centered(RH)
    r_knee = SkeletonNode("RIGHT_KNEE", parent=r_hip, landmark_idx=mp_pose.PoseLandmark.RIGHT_KNEE.value); r_knee.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle = SkeletonNode("RIGHT_ANKLE", parent=r_knee, landmark_idx=mp_pose.PoseLandmark.RIGHT_ANKLE.value); r_ankle.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # Arms (optional for better openpose map)
    l_sh = SkeletonNode("LEFT_SHOULDER", parent=spine, landmark_idx=LS); l_sh.world_pos = get_centered(LS)
    r_sh = SkeletonNode("RIGHT_SHOULDER", parent=spine, landmark_idx=RS); r_sh.world_pos = get_centered(RS)
    l_el = SkeletonNode("LEFT_ELBOW", parent=l_sh, landmark_idx=mp_pose.PoseLandmark.LEFT_ELBOW.value); l_el.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    r_el = SkeletonNode("RIGHT_ELBOW", parent=r_sh, landmark_idx=mp_pose.PoseLandmark.RIGHT_ELBOW.value); r_el.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    l_wr = SkeletonNode("LEFT_WRIST", parent=l_el, landmark_idx=mp_pose.PoseLandmark.LEFT_WRIST.value); l_wr.world_pos = get_centered(mp_pose.PoseLandmark.LEFT_WRIST.value)
    r_wr = SkeletonNode("RIGHT_WRIST", parent=r_el, landmark_idx=mp_pose.PoseLandmark.RIGHT_WRIST.value); r_wr.world_pos = get_centered(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # Fill node_map
    for n in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle, l_sh, r_sh, l_el, r_el, l_wr, r_wr]:
        node_map[n.name] = n

    # Compute local offsets ("rest offsets")
    def compute_local(node: SkeletonNode):
        for ch in node.children:
            ch.local_pos = (ch.world_pos - node.world_pos).astype(np.float32)
            compute_local(ch)
    compute_local(root)

    # Recompute world positions from local offsets (ensure consistency)
    def update_world(node: SkeletonNode):
        for ch in node.children:
            ch.world_pos = node.world_pos + ch.local_pos
            update_world(ch)
    update_world(root)

    return root, node_map

def clone_skeleton(root: SkeletonNode) -> Tuple[SkeletonNode, Dict[str, SkeletonNode]]:
    """Deep clone skeleton tree (only local_pos, world_pos, name, idx)."""
    node_map: Dict[str, SkeletonNode] = {}

    def _clone(node: SkeletonNode, parent: Optional[SkeletonNode]) -> SkeletonNode:
        nn = SkeletonNode(node.name, parent=parent, landmark_idx=node.landmark_idx)
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
    """
    FK: rotate all descendant bone vectors around this joint (preserves bone lengths).
    We rotate each child's local_pos, then recurse.
    """
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
    """
    Diagram-accurate: "Edit pose in 3D using skeletal FK on the tree
    (apply joint rotations, keep bone lengths)".

    This version adds more realistic BAD-form variety using ONLY FK rotations:
      - Forward torso lean
      - Knee valgus (knees collapse in)
      - Knee travel forward (proxy for heel lift / weight on toes)
      - Hip shift / pelvis tilt (asymmetry)
      - Torso twist
      - Pelvic tuck ("butt wink") approximation
    """

    def rng(a, b):
        return random.uniform(a, b)

    # Pick a set of "faults" to apply (variety helps training)
    faults = []
    if random.random() < 0.90: faults.append("forward_lean")
    if random.random() < 0.70: faults.append("knee_valgus")
    if random.random() < 0.45: faults.append("knee_forward")
    if random.random() < 0.40: faults.append("hip_shift")
    if random.random() < 0.30: faults.append("torso_twist")
    if random.random() < 0.30: faults.append("pelvic_tuck")

    # --- Forward lean (torso pitches forward) ---
    if "forward_lean" in faults and "MID_SHOULDER" in node_map:
        # stronger range gives clearer "bad"
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(25, 50)))

        # optionally add slight roll to make it look less "perfectly symmetric"
        if random.random() < 0.35:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(rng(-10, 10)))

    # --- Torso twist (upper body rotated) ---
    if "torso_twist" in faults and "MID_SHOULDER" in node_map:
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_y(rng(-20, 20)))
        apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(rng(-12, 12)))

    # --- Knee valgus (knees collapse inward) ---
    # We rotate knee subtrees around Z in opposite directions (inward collapse)
    if "knee_valgus" in faults:
        if "LEFT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_KNEE"], rot_z(rng(+12, +28)))
        if "RIGHT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_KNEE"], rot_z(rng(-12, -28)))

        # Add a bit of hip internal rotation to amplify valgus (still FK)
        if random.random() < 0.40:
            if "LEFT_HIP" in node_map:
                apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_y(rng(+6, +16)))
            if "RIGHT_HIP" in node_map:
                apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_y(rng(-6, -16)))

    # --- Knee travel forward (proxy for "weight on toes / heels lift") ---
    # We don't have foot nodes, so we approximate by changing shin angle:
    # rotate knee subtrees around X so the ankle moves forward/down in the pose.
    if "knee_forward" in faults:
        # choose a direction per sample to diversify camera conventions
        sgn = 1.0 if random.random() < 0.5 else -1.0
        if "LEFT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_KNEE"], rot_x(sgn * rng(10, 25)))
        if "RIGHT_KNEE" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_KNEE"], rot_x(sgn * rng(10, 25)))

        # Combine with a bit more forward lean often seen when knees shoot forward
        if "MID_SHOULDER" in node_map and random.random() < 0.6:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(8, 18)))

    # --- Hip shift / asymmetry (one side loaded) ---
    # We emulate pelvis tilt by rotating hips in opposite Z directions (still FK),
    # plus slight opposite torso compensation to look "shifted".
    if "hip_shift" in faults:
        shift_dir = 1.0 if random.random() < 0.5 else -1.0
        if "LEFT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_z(shift_dir * rng(6, 16)))
        if "RIGHT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_z(-shift_dir * rng(6, 16)))

        # Often comes with one knee collapsing more
        if "LEFT_KNEE" in node_map and random.random() < 0.5:
            apply_rotation_to_subtree(node_map["LEFT_KNEE"], rot_z(shift_dir * rng(6, 14)))
        if "RIGHT_KNEE" in node_map and random.random() < 0.5:
            apply_rotation_to_subtree(node_map["RIGHT_KNEE"], rot_z(-shift_dir * rng(6, 14)))

        # Upper body compensates a little
        if "MID_SHOULDER" in node_map and random.random() < 0.7:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_z(-shift_dir * rng(4, 10)))

    # --- Pelvic tuck ("butt wink") approximation ---
    # We don't have pelvis/spine multi-joints, so we approximate by
    # rotating both hip subtrees and adding a small extra torso pitch.
    if "pelvic_tuck" in faults:
        if "LEFT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["LEFT_HIP"], rot_x(rng(8, 22)))
        if "RIGHT_HIP" in node_map:
            apply_rotation_to_subtree(node_map["RIGHT_HIP"], rot_x(rng(8, 22)))

        if "MID_SHOULDER" in node_map and random.random() < 0.8:
            apply_rotation_to_subtree(node_map["MID_SHOULDER"], rot_x(rng(6, 16)))

    # Final FK update
    update_world_positions(root)


# -----------------------
# 6) Camera sampling + perspective projection (diagram: sample camera view + intrinsics/extrinsics, normalize to 512 canvas w/ margin)
# -----------------------
def sample_camera_params() -> Dict[str, float]:
    yaw = random.uniform(-45, 45)
    pitch = random.uniform(-12, 12)
    roll = random.uniform(-6, 6)
    distance = random.uniform(3.0, 5.0)
    fov = random.uniform(30, 50)
    return {"yaw": yaw, "pitch": pitch, "roll": roll, "distance": distance, "fov": fov}

def project_3d_to_2d(points_3d: Dict[int, np.ndarray], cam: Dict[str, float],
                    width: int, height: int, margin: float) -> Dict[int, Dict[str, float]]:
    """
    Perspective projection onto normalized [0..1] coords.
    We assume skeleton is centered at root=origin, so "look at origin".
    """
    y = math.radians(cam["yaw"])
    p = math.radians(cam["pitch"])
    r = math.radians(cam["roll"])

    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], dtype=np.float32)
    Rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]], dtype=np.float32)
    Rr = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    R = (Ry @ Rp @ Rr).astype(np.float32)

    # camera position in world coords
    cam_pos = (R @ np.array([0, 0, cam["distance"]], dtype=np.float32)).astype(np.float32)

    f = (height / 2.0) / math.tan(math.radians(cam["fov"]) / 2.0)

    pts2d = {}
    xs, ys = [], []
    for idx, pt in points_3d.items():
        # world -> camera space
        rel = (pt - cam_pos).astype(np.float32)
        cam_pt = (R.T @ rel).astype(np.float32)
        z = -float(cam_pt[2])
        if z < 0.1:
            z = 0.1
        px = (float(cam_pt[0]) * f / z) + (width / 2.0)
        py = (-float(cam_pt[1]) * f / z) + (height / 2.0)
        xs.append(px); ys.append(py)
        pts2d[idx] = (px, py)

    # normalize & fit to canvas with margin
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(1.0, maxx - minx)
    spany = max(1.0, maxy - miny)

    # scale to fit inside (1-2*margin)
    inner_w = width * (1.0 - 2.0 * margin)
    inner_h = height * (1.0 - 2.0 * margin)
    sx = inner_w / spanx
    sy = inner_h / spany
    s = min(sx, sy)

    out: Dict[int, Dict[str, float]] = {}
    for idx, (px, py) in pts2d.items():
        nx = (px - minx) * s + width * margin
        ny = (py - miny) * s + height * margin
        out[idx] = {"x": clamp01(nx / width), "y": clamp01(ny / height)}
    return out

def draw_openpose_like(projected: Dict[int, Dict[str, float]], width: int, height: int) -> Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # lines
    for a, b in mp_pose.POSE_CONNECTIONS:
        if a in projected and b in projected:
            p1 = projected[a]; p2 = projected[b]
            x1, y1 = int(p1["x"] * width), int(p1["y"] * height)
            x2, y2 = int(p2["x"] * width), int(p2["y"] * height)
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 4)

    # joints
    for idx, p in projected.items():
        x, y = int(p["x"] * width), int(p["y"] * height)
        cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)

    return Image.fromarray(canvas)


# -----------------------
# 7) Stage 1: SD Img2Img + ControlNet synthetic dataset generation (diagram-accurate)
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

    # optional speed-up
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe

def _save_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)

def plan_targets(total_images: int) -> Dict[str, Dict[str, int]]:
    """
    Returns targets[split][label] = count
    If TEST_RATIO==0, we still create test targets as 0 and later reuse val for test evaluation.
    """
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
        raise FileNotFoundError("❌ No GOOD seeds found. Put images into seeds/good")

    random.shuffle(good_seeds)
    random.shuffle(bad_seeds)

    # Diagram: split GOOD seeds to train/val (decision diamond)
    n_good = len(good_seeds)
    n_val = max(1, int(round(n_good * cfg.VAL_RATIO))) if n_good >= 2 else 0
    val_good_seeds = good_seeds[-n_val:] if n_val > 0 else []
    train_good_seeds = good_seeds[:-n_val] if n_val > 0 else good_seeds[:]

    # Optional: test split support (if TEST_RATIO>0 we further carve from train_good_seeds)
    test_good_seeds: List[Path] = []
    if cfg.TEST_RATIO > 0 and len(train_good_seeds) >= 3:
        n_test = max(1, int(round(n_good * cfg.TEST_RATIO)))
        test_good_seeds = train_good_seeds[:n_test]
        train_good_seeds = train_good_seeds[n_test:]

    targets = plan_targets(total_images)

    # Count existing (resume-friendly)
    existing = {
        sp: {
            "good": count_images((SYNTH_DIR / sp / "good")),
            "bad":  count_images((SYNTH_DIR / sp / "bad")),
        } for sp in ["train", "val", "test"]
    }
    print("Existing counts:", existing)
    print("Target counts:  ", targets)

    pipe = load_pipe()

    # Cache MP extraction per seed
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
        # Extract + skeleton
        pack = get_pose_and_init(seed_path)
        if pack is None:
            return False

        root_rest, node_map_rest, init_img = pack
        cam = sample_camera_params()

        # Good vs Bad: diagram uses FK edit for bad-from-good; for bad-from-badseeds we use the extracted pose as-is.
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
                image=init_img,                 # Img2Img init image
                control_image=ctrl_img,         # ControlNet conditioning image
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

    # Build generation schedules
    # We want EXACT total_images, so we fill targets precisely.
    # Diagram-accurate sources:
    #  - good_from_goodseeds (train/val/test)
    #  - bad_from_good (train/val/test) via FK edit
    #  - extra_bad_from_badseeds (train only) (optional)
    #
    # To keep TOTAL_IMAGES exact, we allocate train bad quota partially to bad seeds,
    # and the rest to bad_from_good. (counts are configurable)

    # Determine how many train/bad we try to source from bad seeds (within target)
    train_bad_target = targets["train"]["bad"]
    train_bad_from_badseeds = 0
    if cfg.SYNTH_BAD_FROM_BADSEEDS and bad_seeds and train_bad_target > 0:
        train_bad_from_badseeds = int(round(train_bad_target * cfg.BAD_FROM_BADSEEDS_FRACTION_TRAIN))
        train_bad_from_badseeds = max(0, min(train_bad_from_badseeds, train_bad_target))

    # Remaining train bad to generate via bad_from_good
    train_bad_from_good = train_bad_target - train_bad_from_badseeds

    # Use round-robin seed iteration
    def rr(seeds: List[Path]):
        i = 0
        while True:
            if not seeds:
                yield None
            else:
                yield seeds[i % len(seeds)]
                i += 1

    rr_train_good = rr(train_good_seeds)
    rr_val_good   = rr(val_good_seeds if val_good_seeds else train_good_seeds)
    rr_test_good  = rr(test_good_seeds if test_good_seeds else (val_good_seeds if val_good_seeds else train_good_seeds))
    rr_bad        = rr(bad_seeds)

    # Helper to generate until target reached
    def fill_split(split: str, good_target: int, bad_from_good_target: int):
        # 1) generate GOOD from GOOD seeds
        while existing[split]["good"] < good_target:
            seed_path = next(rr_train_good if split == "train" else rr_val_good if split == "val" else rr_test_good)
            if seed_path is None:
                break
            ok = gen_one(split, "good", seed_path, source="good_from_goodseeds")
            if ok:
                existing[split]["good"] += 1

        # 2) generate BAD from GOOD seeds via FK edit
        while existing[split]["bad"] < bad_from_good_target:
            seed_path = next(rr_train_good if split == "train" else rr_val_good if split == "val" else rr_test_good)
            if seed_path is None:
                break
            ok = gen_one(split, "bad", seed_path, source="bad_from_good")
            if ok:
                existing[split]["bad"] += 1

    # TRAIN: good target, bad-from-good target, then optional extra bad-from-badseeds to fill remaining train bad
    fill_split("train", targets["train"]["good"], train_bad_from_good)

    # TRAIN extra: bad seeds (train only)
    while existing["train"]["bad"] < targets["train"]["bad"] and train_bad_from_badseeds > 0:
        seed_path = next(rr_bad)
        if seed_path is None:
            break
        ok = gen_one("train", "bad", seed_path, source="extra_bad_from_badseeds")
        if ok:
            existing["train"]["bad"] += 1
            train_bad_from_badseeds -= 1

    # VAL
    fill_split("val", targets["val"]["good"], targets["val"]["bad"])

    # TEST (if TEST_RATIO==0, targets["test"] likely 0, so it will just skip)
    fill_split("test", targets["test"]["good"], targets["test"]["bad"])

    print("\n✅ Generation complete. Final counts:", {
        sp: {"good": count_images(SYNTH_DIR/sp/"good"), "bad": count_images(SYNTH_DIR/sp/"bad")}
        for sp in ["train", "val", "test"]
    })


# -----------------------
# 8) EDA (diagram request: EDA on the generated dataset)
# -----------------------
def run_eda():
    print("\n=== EDA ===")
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["good", "bad"]:
            paths = list_images(SYNTH_DIR / split / label)
            rows.append({"split": split, "label": label, "count": len(paths)})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "eda_counts.csv", index=False)

    # Plot class balance per split
    pivot = df.pivot(index="split", columns="label", values="count").fillna(0)
    ax = pivot.plot(kind="bar")
    ax.set_title("Class counts per split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_class_counts.png", dpi=150)
    plt.close(fig)

    # Sample montage
    def montage(split: str, label: str, out_name: str, n: int = 12):
        paths = list_images(SYNTH_DIR / split / label)
        if not paths:
            return
        take = paths[:n]
        imgs = [pil_center_crop_resize(safe_imread_pil(p), 224) for p in take]
        cols = 6
        rows = int(math.ceil(len(imgs) / cols))
        canvas = Image.new("RGB", (cols * 224, rows * 224), (0, 0, 0))
        for i, im in enumerate(imgs):
            r = i // cols
            c = i % cols
            canvas.paste(im, (c * 224, r * 224))
        canvas.save(OUTPUT_DIR / out_name)

    montage("train", "good", "eda_samples_train_good.jpg")
    montage("train", "bad", "eda_samples_train_bad.jpg")
    montage("val", "good", "eda_samples_val_good.jpg")
    montage("val", "bad", "eda_samples_val_bad.jpg")

    print("✅ EDA outputs saved to outputs/ (counts CSV + plots + sample montages)")


# -----------------------
# 9) Stage 2: Train ViT-B/16 on synthetic_dataset/train, validate on val (diagram-accurate)
# -----------------------
class SquatDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        # label mapping: bad=0, good=1
        for label_name, y in [("bad", 0), ("good", 1)]:
            for p in list_images(self.root_dir / label_name):
                self.samples.append((p, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = safe_imread_pil(p)
        img = pil_center_crop_resize(img, 224)
        if self.transform:
            img = self.transform(img)
        return img, y, p.name

def train_vit() -> Tuple[Optional[nn.Module], Optional[transforms.Compose]]:
    print("\n=== Stage 2: Train image model (ViT-B/16) ===")

    train_ds = SquatDataset(TRAIN_DIR)
    val_ds = SquatDataset(VAL_DIR)

    if len(train_ds) == 0:
        print("❌ No training data found in synthetic_dataset/train. Skipping training.")
        return None, None

    # torchvision ViT
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    num_f = model.heads.head.in_features
    model.heads.head = nn.Linear(num_f, 2)

    if cfg.FREEZE_VIT_BACKBONE:
        for name, p in model.named_parameters():
            if "heads" not in name:
                p.requires_grad = False

    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds.transform = transform
    val_ds.transform = transform

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)

    for epoch in range(cfg.EPOCHS):
        model.train()
        running = 0.0
        for x, y, _names in train_loader:
            x = x.to(DEVICE)
            y = torch.tensor(y).to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        avg = running / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} - train loss: {avg:.4f}")

    # Save weights
    torch.save(model.state_dict(), OUTPUT_DIR / "vit_squat.pth")
    print("✅ Saved model to outputs/vit_squat.pth")
    return model, transform


# -----------------------
# 10) Evaluation + tables + confusion matrix (diagram: save metrics, confusion matrix, optional predictions)
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

    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

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

            for i in range(len(names)):
                true_y = int(y[i])
                pred_y = int(pred[i])
                conf = float(np.max(probs[i]))
                rows.append({
                    "split": split_name,
                    "image_name": names[i],
                    "true_label": "good" if true_y == 1 else "bad",
                    "pred_label": "good" if pred_y == 1 else "bad",
                    "confidence": conf,
                    "correct": (true_y == pred_y),
                    "p_bad": float(probs[i][0]),
                    "p_good": float(probs[i][1]),
                })
                all_y.append(true_y)
                all_p.append(pred_y)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"{split_name}_predictions.csv", index=False)

    cm = confusion_matrix(all_y, all_p, labels=[0, 1])
    save_confusion_matrix(cm, ["bad", "good"], OUTPUT_DIR / f"confusion_{split_name}.png",
                          title=f"Confusion Matrix ({split_name})")

    rep = classification_report(all_y, all_p, target_names=["bad", "good"], output_dict=True)
    with open(OUTPUT_DIR / f"classification_report_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print(f"\n[{split_name}] Confusion matrix:\n{cm}")
    print(f"[{split_name}] Classification report:\n{classification_report(all_y, all_p, target_names=['bad','good'])}")

    summary = {
        "split": split_name,
        "n": len(df),
        "accuracy": float((df["correct"].mean()) if len(df) else 0.0),
        "confusion_matrix": cm.tolist(),
    }
    return df, summary


# -----------------------
# 11) “LLM feedback” (OpenAI optional; fallback always available)
# -----------------------
def fallback_feedback(row: Dict) -> Dict:
    """
    Rule-based feedback that behaves like an LLM summary:
    - "keep": what was good
    - "improve": what to fix
    """
    true_label = row["true_label"]
    pred_label = row["pred_label"]
    conf = row["confidence"]

    keep = []
    improve = []

    keep.append("Keep consistent camera/lighting style and full-body framing.")
    keep.append("Keep clear visibility of hips, knees, ankles (helps both pose & classifier).")

    if true_label == "good" and pred_label == "bad":
        improve.append("The model likely sees forward torso lean or knee alignment issues—ensure neutral spine and knees tracking toes.")
        improve.append("Increase variety of GOOD samples (angles + body types) to reduce false BAD flags.")
    elif true_label == "bad" and pred_label == "good":
        improve.append("The model may miss subtle knee valgus/torso lean—add more BAD samples emphasizing these faults.")
        improve.append("Add hard negatives: 'almost good but wrong' squats to improve boundary.")
    else:
        improve.append("Add more viewpoint diversity and background variety to improve robustness.")

    summary = f"Prediction={pred_label} (conf={conf:.2f}). Keep: {keep[0]} Improve: {improve[0]}"
    return {
        "generator": "fallback",
        "keep": keep[:2],
        "improve": improve[:2],
        "summary": summary
    }

def openai_feedback_optional(row: Dict) -> Optional[Dict]:
    """
    Optional: if OPENAI_API_KEY is set and openai package exists, use it.
    If not available -> return None (fallback will be used).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import openai  # type: ignore
    except Exception:
        return None

    try:
        # Minimal prompt (no sensitive content; short)
        msg = (
            f"Task: give coaching-style feedback for squat form classification.\n"
            f"True label: {row['true_label']}\nPred label: {row['pred_label']}\nConfidence: {row['confidence']:.2f}\n"
            f"Return JSON with keys: keep (list), improve (list), summary (string)."
        )
        client = openai.OpenAI(api_key=api_key)  # type: ignore
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content": msg}],
            temperature=0.2
        )
        text = resp.choices[0].message.content or ""
        # try parse JSON from response
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        return {
            "generator": "openai",
            "keep": data.get("keep", []),
            "improve": data.get("improve", []),
            "summary": data.get("summary", "")
        }
    except Exception:
        return None

def write_llm_feedback(df: pd.DataFrame, split_name: str, max_items: int = 200):
    """
    Create per-image feedback table (prefer openai if available, otherwise fallback).
    We focus on:
      - all incorrect rows (up to max_items)
      - plus a few lowest-confidence correct rows (to help improve boundary)
    """
    if df is None or df.empty:
        return

    wrong = df[df["correct"] == False].copy()
    low_conf = df[df["correct"] == True].sort_values("confidence").head(max(0, max_items - len(wrong)))

    cand = pd.concat([wrong, low_conf], ignore_index=True)
    cand = cand.head(max_items)

    out_rows = []
    for _, r in cand.iterrows():
        row = r.to_dict()
        fb = openai_feedback_optional(row)
        if fb is None:
            fb = fallback_feedback(row)
        out_rows.append({**row, **fb})

    out_path = OUTPUT_DIR / f"llm_feedback_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    print(f"✅ Feedback saved: {out_path}")


# -----------------------
# 12) README generation (diagram request)
# -----------------------
def write_readme(summaries: List[Dict], note: str = ""):
    lines = []
    lines.append("# PoseAITraining — Run Summary\n")
    lines.append("## Config\n")
    lines.append(f"- Device: {DEVICE}\n")
    lines.append(f"- TOTAL_IMAGES: {cfg.TOTAL_IMAGES}\n")
    lines.append(f"- VAL_RATIO: {cfg.VAL_RATIO}\n")
    lines.append(f"- TEST_RATIO: {cfg.TEST_RATIO}\n")
    lines.append(f"- SD_MODEL: {cfg.SD_MODEL}\n")
    lines.append(f"- CONTROLNET_MODEL: {cfg.CONTROLNET_MODEL}\n")
    lines.append(f"- IMG_SIZE: {cfg.IMG_SIZE}\n")
    lines.append(f"- NUM_STEPS: {cfg.NUM_STEPS}\n")
    lines.append(f"- FREEZE_VIT_BACKBONE: {cfg.FREEZE_VIT_BACKBONE}\n")
    lines.append(f"- EPOCHS: {cfg.EPOCHS}\n")
    lines.append("\n## Outputs\n")
    lines.append("- `synthetic_dataset/` — generated images (train/val/test)\n")
    lines.append("- `outputs/eda_counts.csv` + `outputs/eda_*.png/jpg`\n")
    lines.append("- `outputs/*_predictions.csv` — table with correct/incorrect\n")
    lines.append("- `outputs/confusion_*.png` + `outputs/classification_report_*.json`\n")
    lines.append("- `outputs/llm_feedback_*.json`\n")
    lines.append("- `outputs/vit_squat.pth`\n")
    lines.append("- `outputs/PoseAITraining_artifact.zip`\n")

    if note.strip():
        lines.append("\n## Notes\n")
        lines.append(note.strip() + "\n")

    lines.append("\n## Metrics\n")
    for s in summaries:
        lines.append(f"### {s.get('split','?')}\n")
        lines.append(f"- n = {s.get('n',0)}\n")
        lines.append(f"- accuracy = {s.get('accuracy',0):.4f}\n")

    out_path = OUTPUT_DIR / "README.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ README saved: {out_path}")


# -----------------------
# 13) Zip artifact (download everything: images + code + outputs)
# -----------------------
def zip_artifact():
    zip_path = OUTPUT_DIR / cfg.ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # include synthetic dataset
        if SYNTH_DIR.exists():
            for p in SYNTH_DIR.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(ROOT)))
        # include outputs
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

    print(f"✅ Artifact zip created: {zip_path}")

    # Optional: direct download in Colab
    if _in_colab():
        try:
            from google.colab import files  # type: ignore
            files.download(str(zip_path))
        except Exception:
            pass


# -----------------------
# 14) Main
# -----------------------
def main():
    print(f"ROOT: {ROOT}")
    print(f"DEVICE: {DEVICE}")

    if not GOOD_SEEDS_DIR.exists():
        raise FileNotFoundError("❌ seeds/good folder missing.")
    if not list_images(GOOD_SEEDS_DIR):
        print("❌ Please put seed images into: seeds/good (and optionally seeds/bad)")
        return

    # Stage 1
    if os.environ.get("SKIP_GENERATION", "0") != "1":
        generate_synthetic_dataset(cfg.TOTAL_IMAGES)
    else:
        print("⚠️ SKIP_GENERATION=1 -> skipping generation")

    # EDA
    if os.environ.get("SKIP_EDA", "0") != "1":
        run_eda()
    else:
        print("⚠️ SKIP_EDA=1 -> skipping EDA")

    # Stage 2
    model, transform = (None, None)
    if os.environ.get("SKIP_TRAIN", "0") != "1":
        model, transform = train_vit()
    else:
        print("⚠️ SKIP_TRAIN=1 -> skipping training")

    if model is None or transform is None:
        return

    # Evaluate val
    val_df, val_summary = evaluate_split(model, VAL_DIR, "val", transform)
    write_llm_feedback(val_df, "val")

    # Evaluate test (if empty and TEST_RATIO==0 -> reuse val as test for reporting)
    test_note = ""
    if len(list_images(TEST_DIR / "good")) + len(list_images(TEST_DIR / "bad")) == 0:
        test_note = "TEST split is empty (TEST_RATIO=0). For convenience, test reports reuse val."
        test_df = val_df.copy()
        test_df["split"] = "test"
        test_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
        # copy val confusion/report as test (so you still get the expected files)
        shutil.copyfile(OUTPUT_DIR / "confusion_val.png", OUTPUT_DIR / "confusion_test.png")
        shutil.copyfile(OUTPUT_DIR / "classification_report_val.json", OUTPUT_DIR / "classification_report_test.json")
        test_summary = {"split": "test", "n": int(len(test_df)), "accuracy": float(test_df["correct"].mean() if len(test_df) else 0.0),
                        "confusion_matrix": val_summary.get("confusion_matrix", [])}
        write_llm_feedback(test_df, "test")
    else:
        test_df, test_summary = evaluate_split(model, TEST_DIR, "test", transform)
        write_llm_feedback(test_df, "test")

    # README
    write_readme([val_summary, test_summary], note=test_note)

    # Zip all
    zip_artifact()

    print("\n✅ DONE. Check outputs/ for CSV + confusion matrices + EDA + README + zip artifact.")

if __name__ == "__main__":
    main()
