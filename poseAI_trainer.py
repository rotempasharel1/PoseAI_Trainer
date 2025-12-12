"""
Simple squat demo with:
- Pose estimation (MediaPipe)
- Feature extraction (knee angles, symmetry, hip/ knee height)
- Decision Tree classifier (scikit-learn)
- Optional LLM feedback (OpenAI) for personalized English coaching tips

Pipeline:
1) (Optional) Generate synthetic squat images (good / bad) with Stable Diffusion.
   The text prompts are written to imitate TWO REAL EXAMPLE PHOTOS:
     - images/examples/good_squat_example.jpg  (deep, good squat)
     - images/examples/bad_squat_example.jpg   (shallow / not deep enough squat)
2) From images in images/train/{good,bad}:
       -> run MediaPipe pose
       -> extract numeric features
       -> train a DecisionTreeClassifier.
3) From images in images/test/{good,bad}:
       -> run MediaPipe pose
       -> predict good / bad squat
       -> get coaching tips (LLM if available, otherwise rule-based)
       -> save outputs/poseai_results.csv:
              image_name, is_good, keep_tip, improve_tip
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import csv
import torch
from diffusers import AutoPipelineForText2Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===== NEW: imports for EDA =====
import pandas as pd
import matplotlib.pyplot as plt

# ================== Accuracy threshold ==================
# Minimal required accuracy on the TEST split to consider the model "good enough"
MIN_TEST_ACCURACY = 0.80

# ================== Optional LLM client (OpenAI) ==================

openai_client = None
try:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        print("OpenAI client initialized – LLM feedback enabled.")
    else:
        print("OPENAI_API_KEY is not set – LLM feedback will be disabled.")
except Exception as e:
    print(f"OpenAI client not available ({e}) – LLM feedback will be disabled.")
    openai_client = None

# ================== Paths & folders ==================

IMAGES_DIR = Path("images")
TRAIN_DIR = IMAGES_DIR / "train"
TEST_DIR = IMAGES_DIR / "test"
OUTPUT_DIR = Path("outputs")

for d in [IMAGES_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# תיקיית דוגמאות – כאן שמרת את שתי התמונות מהגוגל
EXAMPLES_DIR = IMAGES_DIR / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

GOOD_SQUAT_EXAMPLE = EXAMPLES_DIR / "good_squat_example.jpg"
BAD_SQUAT_EXAMPLE = EXAMPLES_DIR / "bad_squat_example.jpg"

# train/test subfolders for class labels (used when generating synthetic data)
for base_dir in [TRAIN_DIR, TEST_DIR]:
    (base_dir / "good").mkdir(parents=True, exist_ok=True)
    (base_dir / "bad").mkdir(parents=True, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ================== Geometry helpers ==================

def calculate_angle(a, b, c):
    """
    Angle in degrees at point b for triangle a-b-c.
    a, b, c are (x, y) points.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    return angle


def get_point(landmark, image_shape):
    """Convert MediaPipe landmark to pixel coordinates + visibility."""
    h, w = image_shape[:2]
    return (landmark.x * w, landmark.y * h, landmark.visibility)


# ================== 1) Synthetic image generation (optional) ==================

def build_sd_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """
    Simple Stable Diffusion Text2Image pipeline.
    Uses GPU if available, otherwise CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id).to(device)
    return pipe


def make_squat_prompt(is_good: bool) -> str:
    """
    Create a text prompt that imitates our REAL example photos ONLY.

    We assume:
      - GOOD_SQUAT_EXAMPLE: deep, technically good squat
      - BAD_SQUAT_EXAMPLE : shallow / half squat, not deep enough

    The prompt description is written by looking at those photos:
      - side view
      - same type of clothes
      - same room style (wooden floor, light wall, indoor)
    """

    # Base scene: built to match the Google photos you provided
    base = (
        "ultra realistic DSLR photo, 4k, side view, full body, "
        "fit young woman, dark hair in a bun, wearing a tight black crop top "
        "and black and grey sports leggings, indoor gym room, "
        "wooden floor, light plain wall, natural soft lighting"
    )

    if is_good:
        # Based on good_squat_example.jpg – deep and controlled squat
        pose = (
            "performing a deep bodyweight squat with excellent technique, "
            "hips very low close to the heels, thighs clearly parallel or slightly below parallel to the floor, "
            "knees markedly bent around ninety to one hundred and twenty degrees, "
            "heels flat on the ground, back neutral and straight, "
            "torso slightly inclined forward but chest not collapsed, "
            "arms stretched straight forward at shoulder height for balance, "
            "overall movement looks stable, strong and controlled"
        )
    else:
        # Based on bad_squat_example.jpg – more shallow, not enough depth
        pose = (
            "performing a shallow half-squat with imperfect technique, "
            "hips staying higher, thighs clearly above parallel to the floor, "
            "knees only mildly bent around one hundred and forty to one hundred and seventy degrees, "
            "less hip flexion, movement looks like a small dip instead of a full squat, "
            "arms held forward for balance but the squat depth is visibly insufficient, "
            "overall posture does not reach a proper deep squat position"
        )

    return f"{base}, {pose}"


NEGATIVE_PROMPT = (
    "blurry, distorted, low resolution, jpeg artifacts, watermark, text, logo, "
    "cropped, out of frame, multiple people, crowd, "
    "extra limbs, extra legs, extra arms, extra hands, extra fingers, "
    "multiple knees, fused limbs, disfigured, deformed, mutated, "
    "two heads, extra head, long neck, bad anatomy, bad hands, bad feet, "
    "body out of proportion"
)


def generate_squat_dataset(num_train: int = 7, num_test: int = 3):
    """
    Optional: generate synthetic squat images.
    Saves:
        images/train/good/*.jpg
        images/train/bad/*.jpg
        images/test/good/*.jpg
        images/test/bad/*.jpg

    The GOOD / BAD appearance is defined ONLY according to our two example photos
    and the make_squat_prompt() description above.
    """
    num_total = num_train + num_test
    print(f"Generating {num_total} synthetic images (good/bad squats)...")

    pipe = build_sd_pipeline()
    device = pipe.device

    for i in range(num_total):
        is_good = (i % 2 == 0)
        cls = "good" if is_good else "bad"
        prompt = make_squat_prompt(is_good)

        generator = torch.Generator(device=device.type).manual_seed(1234 + i)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=6.5,
            height=512,
            width=512,
            generator=generator,
        ).images[0]

        target_root = TRAIN_DIR if i < num_train else TEST_DIR
        out_path = target_root / cls / f"{cls}_squat_{i:03d}.jpg"
        image.save(out_path)
        print(f"Saved {out_path}")

    print("Done generating images.")


# ================== 2) Feature extraction ==================

def extract_features_and_stats(pose_landmarks, image_shape):
    """
    Extract numeric features from pose landmarks for ML:

    Features:
        - knee_min         : minimum of left/right knee angles (more bent leg)
        - knee_max         : maximum of left/right knee angles
        - angle_diff       : |left_angle - right_angle|
        - hip_height_ratio : how low the hip is between shoulder and ankle (0..1)
        - knee_height_diff : normalized |y_left_knee - y_right_knee|

    Also returns a stats dict for human-readable / LLM tips.
    """
    lm = pose_landmarks.landmark
    h, w = image_shape[:2]

    # Left side
    left_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP.value], image_shape)
    left_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE.value], image_shape)
    left_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], image_shape)
    left_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], image_shape)

    # Right side
    right_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP.value], image_shape)
    right_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], image_shape)
    right_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], image_shape)
    right_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], image_shape)

    left_angle = calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
    right_angle = calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])

    knee_min = min(left_angle, right_angle)
    knee_max = max(left_angle, right_angle)
    angle_diff = abs(left_angle - right_angle)

    # choose side of more bent leg (for hip/shoulder reference)
    if left_angle <= right_angle:
        side = "left"
        hip_y = left_hip[1]
        ankle_y = left_ankle[1]
        shoulder_y = left_shoulder[1]
    else:
        side = "right"
        hip_y = right_hip[1]
        ankle_y = right_ankle[1]
        shoulder_y = right_shoulder[1]

    # hip_height_ratio in [0,1]: 0 ≈ standing tall, 1 ≈ very low
    denom = (ankle_y - shoulder_y) + 1e-6
    hip_height_ratio = (hip_y - shoulder_y) / denom

    # knee height difference (0 = same height, big = one knee much lower/higher)
    knee_height_diff = abs(left_knee[1] - right_knee[1]) / float(h)

    features = np.array(
        [knee_min, knee_max, angle_diff, hip_height_ratio, knee_height_diff],
        dtype=np.float32,
    )

    stats = {
        "side_used": side,
        "knee_min": knee_min,
        "knee_max": knee_max,
        "angle_diff": angle_diff,
        "hip_height_ratio": hip_height_ratio,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "knee_height_diff": knee_height_diff,
    }

    return features, stats


# ================== 3) Train Decision Tree ==================

def train_decision_tree():
    """
    Train a DecisionTreeClassifier using images in images/train/good and images/train/bad.
    """
    image_files = list(TRAIN_DIR.rglob("*.jpg")) + list(TRAIN_DIR.rglob("*.png"))

    if not image_files:
        print(f"No training images found in {TRAIN_DIR}. Cannot train decision tree.")
        return None

    X, y = [], []

    with mp_pose.Pose(static_image_mode=True) as pose:
        for img_path in image_files:
            path_lower = str(img_path.parent).lower()
            if "good" in path_lower:
                label = 1
            elif "bad" in path_lower:
                label = 0
            else:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if not results.pose_landmarks:
                continue

            features, _ = extract_features_and_stats(results.pose_landmarks, image.shape)
            X.append(features)
            y.append(label)

    if len(X) < 2 or len(set(y)) < 2:
        print("Not enough labeled training samples to train a decision tree.")
        return None

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    train_pred = clf.predict(X)
    acc = accuracy_score(y, train_pred)
    print(f"Decision tree trained on {len(X)} images. Train accuracy: {acc:.2f}")

    return clf


# ================== 4) LLM feedback (optional) ==================

def generate_llm_feedback(is_good: bool, stats: dict) -> tuple[str, str]:
    """
    Use an LLM to generate short English feedback:
      - keep_tip   : what the trainee is already doing well
      - improve_tip: what they should improve
    """
    if openai_client is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client or API key not available")

    description = (
        f"knee_min={stats['knee_min']:.1f}°, "
        f"knee_max={stats['knee_max']:.1f}°, "
        f"angle_diff={stats['angle_diff']:.1f}°, "
        f"hip_height_ratio={stats['hip_height_ratio']:.2f}, "
        f"knee_height_diff={stats['knee_height_diff']:.2f}"
    )

    user_msg = (
        "You are a fitness coach. A model analyzed a squat from a single image and "
        f"produced these numeric features: {description}. "
        f"The automatic classifier decided that this squat is {'GOOD' if is_good else 'NOT GOOD'}.\n\n"
        "Please return a very short JSON object in English with exactly two fields:\n"
        '  keep_tip   - one short sentence about what is already good in the squat (or "" if nothing).\n'
        '  improve_tip- one short sentence about how to improve (or "" if squat is already good).\n'
        "Do not add any explanations. Only return valid JSON."
    )

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise fitness coach generating very short English feedback.",
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=120,
    )

    text = resp.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        keep_tip = str(data.get("keep_tip", "")).strip()
        improve_tip = str(data.get("improve_tip", "")).strip()
    except Exception:
        keep_tip = "" if not is_good else "Good squat technique – keep controlling the movement."
        improve_tip = text if not is_good else ""

    return keep_tip, improve_tip


# ================== 5) Evaluate single image ==================

def evaluate_squat_dt(pose_landmarks, image_shape, clf: DecisionTreeClassifier | None):
    """
    Use the trained decision tree + manual override to classify squat quality.
    Also generate short English keep / improve tips (LLM if possible, otherwise rule-based).
    """
    features, stats = extract_features_and_stats(pose_landmarks, image_shape)

    # --- prediction from tree (if exists) ---
    if clf is not None:
        pred = clf.predict(features.reshape(1, -1))[0]
        is_good_tree = bool(pred == 1)
    else:
        is_good_tree = False

    # --- manual rule override (designed to catch proper deep squats) ---
    knee_min = stats["knee_min"]
    knee_max = stats["knee_max"]
    angle_diff = stats["angle_diff"]
    hip_ratio = stats["hip_height_ratio"]
    knee_height_diff = stats["knee_height_diff"]

    manual_good = (
        70.0 <= knee_min <= 140.0 and
        knee_max <= 165.0 and
        angle_diff <= 55.0 and
        0.40 <= hip_ratio <= 0.95 and
        knee_height_diff <= 0.18
    )

    is_good = is_good_tree or manual_good

    # --- coaching tips (LLM first, then fallback) ---
    keep_tip = ""
    improve_tip = ""

    try:
        keep_tip, improve_tip = generate_llm_feedback(is_good, stats)
    except Exception as e:
        print(f"LLM feedback failed or not available: {e}")
        if is_good:
            keep_tip = (
                "Great squat depth and symmetry – keep your chest up and your weight on mid-foot and heels."
            )
            improve_tip = ""
        else:
            if knee_min >= 150:
                improve_tip = "Bend your knees more and sit your hips back – go lower into the squat."
            elif knee_min <= 60:
                improve_tip = "You are squatting very deep – stop a bit higher to protect your knees."
            elif angle_diff >= 55:
                improve_tip = "Try to bend both knees to a similar depth – avoid turning it into a lunge."
            elif hip_ratio < 0.4:
                improve_tip = "Lower your hips more while keeping your heels on the floor."
            else:
                improve_tip = "Keep your chest up, brace your core, and move down and up with control."

    return {
        "side_used": stats["side_used"],
        "knee_angle": knee_min,
        "pred_is_good": is_good,
        "keep_tip": keep_tip,
        "improve_tip": improve_tip,
    }


def analyze_image(image_path: Path, clf: DecisionTreeClassifier | None):
    """Run MediaPipe on a single image, draw skeleton + verdict, return decision-tree result."""
    print(f"\nAnalyzing {image_path.name}...")

    image = cv2.imread(str(image_path))
    if image is None:
        print("Could not load image.")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No person detected.")
        return None

    eval_result = evaluate_squat_dt(results.pose_landmarks, image.shape, clf)

    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
    )

    verdict = "Good squat" if eval_result["pred_is_good"] else "Needs improvement"
    color = (0, 200, 0) if eval_result["pred_is_good"] else (0, 0, 255)

    cv2.putText(
        annotated,
        verdict,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        2,
        cv2.LINE_AA,
    )

    out_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"Annotated image saved to: {out_path}")
    print(f"Knee angle (more bent leg): {eval_result['knee_angle']:.1f}° ({verdict})")

    return eval_result


# ================== 6) Run on a folder and save CSV ==================

def run_demo_on_folder(split: str, clf: DecisionTreeClassifier | None):
    """
    Go over images in images/<split>/good and images/<split>/bad,
    run the analysis, and save a CSV with:
        image_name, is_good, keep_tip, improve_tip
    """
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    image_files = list(base_dir.rglob("*.jpg")) + list(base_dir.rglob("*.png"))

    if not image_files:
        print(f"No images found in {base_dir}")
        return

    csv_path = OUTPUT_DIR / "poseai_results.csv"
    header = ["image_name", "is_good", "keep_tip", "improve_tip"]

    y_true, y_pred = [], []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_path in image_files:
            path_lower = str(img_path.parent).lower()
            if "good" in path_lower:
                true_label = 1
            elif "bad" in path_lower:
                true_label = 0
            else:
                true_label = None

            result = analyze_image(img_path, clf)
            if result is None:
                continue

            pred_label = int(result["pred_is_good"])

            writer.writerow([
                img_path.name,
                pred_label,
                result["keep_tip"],
                result["improve_tip"],
            ])

            if true_label is not None:
                y_true.append(true_label)
                y_pred.append(pred_label)

    print(f"\nCSV saved to: {csv_path}")
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall accuracy on {split} split: {acc:.2f}")

        # ✅ Check against the required minimal accuracy – only on TEST split
        if split == "test":
            if acc < MIN_TEST_ACCURACY:
                print(
                    f"WARNING: test accuracy ({acc:.2f}) is BELOW the required threshold "
                    f"({MIN_TEST_ACCURACY:.2f}). The model is not accurate enough."
                )
            else:
                print(
                    f"GOOD: test accuracy ({acc:.2f}) is ABOVE the required threshold "
                    f"({MIN_TEST_ACCURACY:.2f})."
                )


# ================== 7) BASIC EDA FUNCTIONS ==================

def build_feature_dataframe(split: str = "train") -> pd.DataFrame:
    """
    Build a DataFrame with numeric pose features + labels
    for all images in images/<split>/{good,bad}.

    Columns:
      image_name, label, label_name,
      knee_min, knee_max, angle_diff, hip_height_ratio, knee_height_diff
    """
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    image_files = list(base_dir.rglob("*.jpg")) + list(base_dir.rglob("*.png"))

    rows = []

    if not image_files:
        print(f"[EDA] No images found in {base_dir}")
        return pd.DataFrame()

    with mp_pose.Pose(static_image_mode=True) as pose:
        for img_path in image_files:
            path_lower = str(img_path.parent).lower()
            if "good" in path_lower:
                label = 1
                label_name = "good"
            elif "bad" in path_lower:
                label = 0
                label_name = "bad"
            else:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if not results.pose_landmarks:
                continue

            features, stats = extract_features_and_stats(results.pose_landmarks, image.shape)

            rows.append({
                "image_name": img_path.name,
                "label": label,
                "label_name": label_name,
                "knee_min": stats["knee_min"],
                "knee_max": stats["knee_max"],
                "angle_diff": stats["angle_diff"],
                "hip_height_ratio": stats["hip_height_ratio"],
                "knee_height_diff": stats["knee_height_diff"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[EDA] No rows created – maybe no valid poses were detected.")
    else:
        out_path = OUTPUT_DIR / f"features_{split}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[EDA] Features DataFrame for split '{split}' saved to: {out_path}")

    return df


def run_basic_eda(split: str = "train"):
    """
    Run simple EDA:
      - check basic info
      - describe numeric columns
      - class balance (good/bad)
      - histograms per feature
      - boxplots per label for each feature
      - correlation matrix
    Saves plots to outputs/ folder.
    """
    print(f"\n[EDA] Running basic EDA on split: {split}")
    df = build_feature_dataframe(split=split)
    if df.empty:
        print("[EDA] Aborting EDA – empty DataFrame.")
        return

    # Basic info
    print("\n[EDA] Head of DataFrame:")
    print(df.head())

    print("\n[EDA] Class balance (label_name):")
    print(df["label_name"].value_counts())

    print("\n[EDA] Numeric describe:")
    print(df[["knee_min", "knee_max", "angle_diff",
              "hip_height_ratio", "knee_height_diff"]].describe())

    # Histograms for numeric features
    hist_path = OUTPUT_DIR / f"eda_histograms_{split}.png"
    plt.figure(figsize=(12, 8))
    df[["knee_min", "knee_max", "angle_diff",
        "hip_height_ratio", "knee_height_diff"]].hist(bins=15, layout=(2, 3), figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"[EDA] Histograms saved to: {hist_path}")

    # Boxplots by label_name
    boxplot_path = OUTPUT_DIR / f"eda_boxplots_{split}.png"
    features = ["knee_min", "knee_max", "angle_diff",
                "hip_height_ratio", "knee_height_diff"]

    plt.figure(figsize=(14, 8))
    for i, col in enumerate(features, start=1):
        plt.subplot(2, 3, i)
        df.boxplot(column=col, by="label_name")
        plt.title(col)
        plt.xlabel("label_name")
    plt.suptitle(f"Feature distributions by label ({split} split)")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[EDA] Boxplots saved to: {boxplot_path}")

    # Correlation matrix
    corr_path = OUTPUT_DIR / f"eda_corr_{split}.png"
    corr = df[features].corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.yticks(range(len(features)), features)
    plt.title(f"Correlation matrix ({split} split)")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    print(f"[EDA] Correlation matrix saved to: {corr_path}")


# ================== Main ==================

if __name__ == "__main__":
    # OPTIONAL: first time, you can generate synthetic data:
    # generate_squat_dataset(num_train=100, num_test=20)

    print("Training decision tree on train split...")
    clf = train_decision_tree()
    if clf is None:
        print("Warning: decision tree was not trained – falling back to manual rules only.")

    # Run EDA on TRAIN split to show data quality & distributions
    run_basic_eda(split="train")

    print("Running demo on test split...")
    run_demo_on_folder(split="test", clf=clf)
