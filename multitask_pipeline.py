# src/multitask_pipeline.py
import numpy as np
import cv2

from config import DRIVER_SELECT
from fatigue_metrics import eye_aspect_ratio, mouth_aspect_ratio


def _select_driver_face(face_boxes_xyxy: np.ndarray, mode="max_x2"):
    if face_boxes_xyxy is None or len(face_boxes_xyxy) == 0:
        return None
    if mode == "max_x2":
        idx = int(np.argmax(face_boxes_xyxy[:, 2]))
    else:
        areas = (face_boxes_xyxy[:, 2] - face_boxes_xyxy[:, 0]) * (face_boxes_xyxy[:, 3] - face_boxes_xyxy[:, 1])
        idx = int(np.argmax(areas))
    return face_boxes_xyxy[idx]


def _safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w,     x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h,     y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def frametest(yolo_backend, pfld_model, frame_bgr):
    """
    每帧输出（不做perclos最终判定）：
    ret = (lab, eye, mouth, dets, kps_frame)

    lab: list[str]  命中的分心类标签（'phone','smoke','drink'）
    eye: float|None EAR
    mouth: float|None MAR
    dets: list[dict] yolo原始输出（用于画框/可视化）
    kps_frame: np.ndarray|None, shape=(106,2)，映射到原图坐标系的PFLD关键点
    """
    dets = yolo_backend.infer(frame_bgr)

    # 分心标签（统一小写）
    lab = []
    for d in dets:
        name = str(d.get("name", "")).lower()
        if name in ("phone", "smoke", "drink"):
            lab.append(name)

    # 提取face框
    face_xyxy = []
    for d in dets:
        name = str(d.get("name", "")).lower()
        if name == "face":
            face_xyxy.append(np.array(d["xyxy"], dtype=np.float32))
    if len(face_xyxy) == 0:
        return (lab, None, None, dets, None), frame_bgr

    face_xyxy = np.stack(face_xyxy, axis=0)
    driver_box = _select_driver_face(face_xyxy, mode=DRIVER_SELECT)
    if driver_box is None:
        return (lab, None, None, dets, None), frame_bgr

    x1, y1, x2, y2 = driver_box[:4]
    crop = _safe_crop(frame_bgr, x1, y1, x2, y2)
    if crop is None:
        return (lab, None, None, dets, None), frame_bgr

    face_112 = cv2.resize(crop, (112, 112))
    kps = pfld_model.infer(face_112)  # (106,2) in 112x112

    # 将112坐标系关键点映射回原图，便于在主画面可视化
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    kps_frame = np.empty_like(kps, dtype=np.float32)
    kps_frame[:, 0] = kps[:, 0] * (bw / 112.0) + float(x1)
    kps_frame[:, 1] = kps[:, 1] * (bh / 112.0) + float(y1)

    eye = eye_aspect_ratio(kps)
    mouth = mouth_aspect_ratio(kps)

    return (lab, float(eye), float(mouth), dets, kps_frame), frame_bgr