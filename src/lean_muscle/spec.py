"""Locked product numbers. Change these only when the scoring contract changes."""

from __future__ import annotations

GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"

POSE_MODEL = "usyd-community/vitpose-plus-large"
SEGMENT_MODEL = "facebook/sam3.1"
SEGMENT_PROMPT = "person"
SUMMARY_MODEL = "qwen/qwen3.8-27b"
SUMMARY_TEMPERATURE = 0.7

# COCO-17, the order ViTPose returns on this gateway.
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

TORSO_JOINTS = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
)
ELBOW_JOINTS = ("left_elbow", "right_elbow")
# Shoulders and hips define the torso. A few percent past the border is
# rounding; farther than this, the width is not in the photo.
TORSO_FRAME_MARGIN = 0.03
# Elbows may sit outside the photo. They still aim the arm mark, and that
# arm is still scored. Wrists are never required.
JOINT_LABELS = {
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_hip": "左髋",
    "right_hip": "右髋",
}

MIN_KPT_SCORE = 0.3
SECOND_PERSON_AREA_RATIO = 0.6
MAX_IMAGE_SIDE = 1280
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Shoulder width / waist width. Clamped at the ends.
SHOULDER_KNOTS = ((1.05, 60.0), (1.20, 80.0), (1.35, 92.0), (1.50, 98.0))
# Chest width / waist width, from the front.
CHEST_KNOTS = ((1.05, 60.0), (1.20, 80.0), (1.35, 90.0), (1.50, 97.0))
# Lat flare / waist width. A front photo cannot see the back; this is the
# width under the armpit, where the lats show.
BACK_KNOTS = ((1.02, 48.0), (1.15, 72.0), (1.28, 86.0), (1.42, 95.0))
# Vertical edges across the chest: none, one, two, three or more.
CHEST_DEFINITION_SCORES = {0: 40.0, 1: 62.0, 2: 80.0, 3: 92.0}
# Mean arm width / waist width.
ARM_KNOTS = ((0.18, 70.0), (0.24, 90.0), (0.30, 92.0), (0.38, 75.0))
# Used when the upper arm is out of frame or stuck to the torso.
ARM_HIDDEN_SCORE = 42.0
# Checked from the larger threshold down.
ARM_ASYMMETRY_PENALTIES = ((0.25, 25.0), (0.15, 15.0))

ABS_GROOVE_SCORES = {0: 40.0, 1: 55.0, 2: 75.0, 3: 88.0, 4: 95.0}
# min/max horizontal response. At or above FULL, no discount.
# At or below HALF ("差过一半"), multiply by FACTOR.
# Between them, linear. Zero grooves skip the multiplier.
ABS_LR_FULL = 0.7
ABS_LR_HALF = 0.5
ABS_LR_FACTOR = 0.6

# Abs, chest, and the shoulder-to-waist ratio are the core. Back and arms still count.
WEIGHT_ABS = 0.35
WEIGHT_CHEST = 0.25
WEIGHT_SHOULDER = 0.25
WEIGHT_BACK = 0.08
WEIGHT_ARM = 0.07

SENTENCE_MAX_CHARS = 40
# A perpendicular arm chord longer than this fraction of shoulder width
# is the arm and the torso sharing one mask run.
ARM_STUCK_FRACTION = 0.45

SCORE_WEIGHTS = {
    "abs": WEIGHT_ABS,
    "chest": WEIGHT_CHEST,
    "shoulder": WEIGHT_SHOULDER,
    "back": WEIGHT_BACK,
    "arm": WEIGHT_ARM,
}
