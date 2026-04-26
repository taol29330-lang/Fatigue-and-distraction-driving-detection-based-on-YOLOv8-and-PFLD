# src/config.py
import numpy as np

# -------- Camera calibration (YOUR REAL CALIBRATION) --------
CAM_K = np.array([
    [1.16172904e+03, 0.0,           9.41078576e+02],
    [0.0,            1.16856599e+03, 5.33941437e+02],
    [0.0,            0.0,            1.0],
], dtype=np.float32)

CAM_D = np.array([0.21214358, -0.35059664, -0.00200202, -0.00199817, 0.18001655],
                 dtype=np.float32).reshape(5, 1)

# -------- Choose YOLO backend --------
# "ultralytics" for .pt; "mindx_om" for MindX .om
YOLO_BACKEND = "ultralytics"

YOLO_PT_PATH = "./weights/best.pt"
YOLO_OM_PATH = "weights/model.om"

# class names (your 4 classes)
CLASS_NAMES = ["Face", "Smoke", "Phone", "Drink"]

# -------- PFLD --------
PFLD_ONNX_PATH = "./weights/pfld_106.onnx"

# -------- Runtime --------
CAMERA_ID = 0
IMG_SIZE = 640
YOLO_CONF = 0.6
YOLO_IOU = 0.45

# -------- Driver face selection --------
DRIVER_SELECT = "max_x2"   # or "largest_area"

# -------- Fatigue thresholds (need your real tuning) --------
EAR_THRESH = 0.203
MAR_THRESH = 0.382
PERCL0S_THRESH = 0.38


WINDOW_FRAMES = 15
FATIGUE_RATIO = 0.4
WARNING_SCORE_THRESH = 3

# -------- Distraction decision --------
DISTRACT_CLASSES = {"Smoke", "Phone", "Drink"}
DISTRACT_MIN_HITS = 3          # 连续/窗口内命中次数触发
DISTRACT_WINDOW_FRAMES = 15