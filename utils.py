import numpy as np
import cv2

def select_driver_face(face_boxes_xyxy: np.ndarray, mode="max_x2"):
    if face_boxes_xyxy is None or len(face_boxes_xyxy) == 0:
        return None
    boxes = face_boxes_xyxy
    if mode == "max_x2":
        idx = int(np.argmax(boxes[:, 2]))
    elif mode == "largest_area":
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = 0
    return boxes[idx]

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w,     x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h,     y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def draw_box(img, box, color=(50, 50, 250), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_keypoints(face_img, kps_112, color=(0,255,0), radius=1):
    vis = face_img.copy()
    for i, (x, y) in enumerate(kps_112.astype(int)):
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis