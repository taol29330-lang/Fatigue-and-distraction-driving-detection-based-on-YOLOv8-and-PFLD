import argparse
import time
from collections import deque

import cv2
import numpy as np

from config import (
    CAMERA_ID,
    PFLD_ONNX_PATH,
    YOLO_BACKEND,
    YOLO_CONF,
    YOLO_OM_PATH,
    YOLO_PT_PATH,
    CLASS_NAMES,
)
from multitask_pipeline import frametest
from pfld_onnx import Pfld106
from yolo_ultralytics import YoloUltralytics

try:
    from yolo_mindx import YoloMindXOM
except Exception:
    YoloMindXOM = None


def build_yolo_backend():
    if YOLO_BACKEND == "ultralytics":
        return YoloUltralytics(YOLO_PT_PATH, conf=YOLO_CONF)
    if YOLO_BACKEND == "mindx_om":
        if YoloMindXOM is None:
            raise RuntimeError("YOLO_BACKEND=mindx_om, but yolo_mindx.py is unavailable")
        return YoloMindXOM(YOLO_OM_PATH, class_names=CLASS_NAMES, imgsz=640, conf=0.6, iou=0.45)
    raise RuntimeError(f"Unsupported YOLO_BACKEND: {YOLO_BACKEND}")


def summarize(values, name):
    arr = np.asarray(values, dtype=np.float32)
    return {
        "name": name,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_stats(st):
    print(f"{st['name']} count={st['count']}")
    print(
        f"  mean={st['mean']:.4f} std={st['std']:.4f} "
        f"p10={st['p10']:.4f} p50={st['p50']:.4f} p90={st['p90']:.4f} "
        f"min={st['min']:.4f} max={st['max']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="采集正常状态下的 EAR/MAR 基线")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="摄像头ID")
    parser.add_argument("--frames", type=int, default=300, help="采集帧数")
    parser.add_argument("--warmup", type=int, default=30, help="预热帧数")
    parser.add_argument("--show", action="store_true", help="显示采集窗口")
    args = parser.parse_args()

    yolo_backend = build_yolo_backend()
    pfld_model = Pfld106(PFLD_ONNX_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("请保持正常驾驶姿态（睁眼、闭嘴、正视前方）进行采集...")
    print(f"预热 {args.warmup} 帧后，采集 {args.frames} 帧。")

    ears = []
    mars = []
    perclos_series = []
    eye_closed_win = deque(maxlen=150)
    mouth_open_win = deque(maxlen=150)
    frame_idx = 0
    valid_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx <= args.warmup:
            if args.show:
                cv2.putText(frame, "Warming up...", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("baseline_ear_mar", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
            continue

        (lab, eye, mouth, dets, kps_frame), vis = frametest(yolo_backend, pfld_model, frame)
        if eye is not None and mouth is not None:
            ears.append(float(eye))
            mars.append(float(mouth))
            valid_idx += 1

            eye_closed_win.append(1 if eye < 0.20 else 0)
            mouth_open_win.append(1 if mouth > 0.60 else 0)
            if len(eye_closed_win) == 150:
                p = (sum(eye_closed_win) / 150.0) + (sum(mouth_open_win) / 150.0) * 0.2
                perclos_series.append(float(p))

        if args.show:
            text = f"valid={valid_idx}/{args.frames} EAR={0.0 if eye is None else eye:.3f} MAR={0.0 if mouth is None else mouth:.3f}"
            cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("baseline_ear_mar", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        if valid_idx >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n采集结束，用时 {elapsed:.1f}s，拿到有效样本 {len(ears)} 帧")

    if len(ears) < max(30, args.frames // 3):
        print("有效样本偏少，请检查光照/脸部角度后重试。")
        return

    ear_stats = summarize(ears, "EAR")
    mar_stats = summarize(mars, "MAR")
    perclos_stats = summarize(perclos_series, "PERCLOS") if len(perclos_series) > 0 else None

    print("\n=== 正常状态统计 ===")
    print_stats(ear_stats)
    print_stats(mar_stats)
    if perclos_stats is not None:
        print_stats(perclos_stats)

    # 建议阈值：EAR 用较低分位数下移，MAR 用较高分位数上移
    ear_thresh_suggest = max(0.05, ear_stats["p10"] * 0.90)
    mar_thresh_suggest = mar_stats["p90"] * 1.10
    perclos_thresh_suggest = max(0.05, perclos_stats["p90"] * 1.10) if perclos_stats is not None else 0.38

    print("\n=== 阈值建议（可先试用再微调）===")
    print(f"EAR_THRESH 建议: {ear_thresh_suggest:.3f}")
    print(f"MAR_THRESH 建议: {mar_thresh_suggest:.3f}")
    print(f"PERCL0S_THRESH 建议: {perclos_thresh_suggest:.3f}")


if __name__ == "__main__":
    main()
