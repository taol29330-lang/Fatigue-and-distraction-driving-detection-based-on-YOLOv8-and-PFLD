import time
from collections import deque

import cv2
from PySide6.QtCore import QThread, Signal

from config import (
    CAMERA_ID,
    YOLO_BACKEND, YOLO_PT_PATH, YOLO_OM_PATH, CLASS_NAMES, YOLO_CONF,
    PFLD_ONNX_PATH,
    EAR_THRESH, MAR_THRESH, PERCL0S_THRESH,
)
from fatigue_metrics import IDX as FATIGUE_KP_IDX

from pfld_onnx import Pfld106
from multitask_pipeline import frametest

from yolo_ultralytics import YoloUltralytics
try:
    from yolo_mindx import YoloMindXOM
except Exception:
    YoloMindXOM = None


class CameraWorker(QThread):
    frame_signal = Signal(object)   # np.ndarray BGR frame
    status_signal = Signal(dict)    # status dict
    log_signal = Signal(str)        # log text

    def __init__(self, camera_id=CAMERA_ID, ear_thresh=EAR_THRESH,
                 mar_thresh=MAR_THRESH, perclos_thresh=PERCL0S_THRESH,
                 parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.ear_thresh = float(ear_thresh)
        self.mar_thresh = float(mar_thresh)
        self.perclos_thresh = float(perclos_thresh)
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True

        FPS_CAP = 20  # FPS显示上限

        # --- init YOLO backend ---
        if YOLO_BACKEND == "ultralytics":
            yolo = YoloUltralytics(YOLO_PT_PATH, conf=YOLO_CONF)
            self.log_signal.emit(f"YOLO backend: ultralytics ({YOLO_PT_PATH})")
        elif YOLO_BACKEND == "mindx_om":
            if YoloMindXOM is None:
                self.log_signal.emit("YOLO_BACKEND=mindx_om but yolo_mindx.py import failed.")
                return
            yolo = YoloMindXOM(YOLO_OM_PATH, class_names=CLASS_NAMES, imgsz=640, conf=0.6, iou=0.45)
            self.log_signal.emit(f"YOLO backend: mindx_om ({YOLO_OM_PATH})")
        else:
            self.log_signal.emit(f"Unknown YOLO_BACKEND={YOLO_BACKEND}")
            return

        # --- init PFLD ---
        pfld = Pfld106(PFLD_ONNX_PATH)

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.log_signal.emit(f"Cannot open camera id={self.camera_id}")
            return

        # FPS (moving avg)
        fps_hist = deque(maxlen=26)
        last_t = time.perf_counter()

        # ====== distraction counters (as your reference) ======
        ActionCOUNTER = 0  # 计数到15则复位UI

        # ====== fatigue counters (as your reference) ======
        EYE_AR_CONSEC_FRAMES = 3
        MOUTH_AR_CONSEC_FRAMES = 3

        COUNTER = 0
        TOTAL = 0
        mCOUNTER = 0
        mTOTAL = 0

        PERCL0S_WINDOW = 150
        eye_closed_win = deque(maxlen=PERCL0S_WINDOW)
        mouth_open_win = deque(maxlen=PERCL0S_WINDOW)
        perclos_log_counter = 0
        last_fatigue_alarm = None

        self.log_signal.emit(
            f"Fatigue thresholds: EAR<{self.ear_thresh:.3f}, "
            f"MAR>{self.mar_thresh:.3f}, PERCLOS>{self.perclos_thresh:.3f}"
        )

        # 为了避免“请不要分心”刷屏：边沿触发
        last_distract_any = False

        self.log_signal.emit("Camera opened, start processing...")

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.log_signal.emit("Camera read failed.")
                break

            # FPS
            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            if dt > 0:
                fps_hist.append(1.0 / dt)
            fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            fps = min(fps, FPS_CAP)  # cap to 27

            # Inference: ret=(lab, eye, mouth, dets, kps_frame)
            (lab, eye, mouth, dets, kps_frame), frame = frametest(yolo, pfld, frame)

            # --- draw YOLO boxes ---
            hit_names_lower = set()
            for d in dets:
                name = str(d.get("name", ""))
                hit_names_lower.add(name.lower())
                conf = float(d.get("conf", 0.0))
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                color = (0, 255, 0) if name.lower() == "face" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 仅绘制“参与计算”的PFLD关键点，避免106点全部显示过于拥挤
            if kps_frame is not None and len(kps_frame) > 0:
                used_idx = set([
                    FATIGUE_KP_IDX["le_l"], FATIGUE_KP_IDX["le_r"],
                    FATIGUE_KP_IDX["re_l"], FATIGUE_KP_IDX["re_r"],
                    FATIGUE_KP_IDX["m_l"], FATIGUE_KP_IDX["m_r"],
                ])
                used_idx.update(FATIGUE_KP_IDX["le_top"])
                used_idx.update(FATIGUE_KP_IDX["le_bot"])
                used_idx.update(FATIGUE_KP_IDX["re_top"])
                used_idx.update(FATIGUE_KP_IDX["re_bot"])
                used_idx.update(FATIGUE_KP_IDX["m_top"])
                used_idx.update(FATIGUE_KP_IDX["m_bot"])

                for idx in used_idx:
                    if 0 <= idx < len(kps_frame):
                        px, py = kps_frame[idx]
                        cv2.circle(frame, (int(px), int(py)), 2, (255, 220, 0), -1)

            # ====== distraction logic (ActionCOUNTER like your code) ======
            ActionCOUNTER += 1
            phone_hit = ("phone" in lab)
            smoke_hit = ("smoke" in lab)
            drink_hit = ("drink" in lab)

            if phone_hit or smoke_hit or drink_hit:
                if ActionCOUNTER > 0:
                    ActionCOUNTER -= 1

            if ActionCOUNTER >= 15:
                phone_hit = smoke_hit = drink_hit = False
                ActionCOUNTER = 0

            distract_any = bool(phone_hit or smoke_hit or drink_hit)

            # UI日志：检测到分心时打印一次“请不要分心”
            if distract_any and not last_distract_any:
                self.log_signal.emit("请不要分心")
            last_distract_any = distract_any

            # ====== fatigue counters update (NO per-frame final judgement) ======
            if eye is not None:
                if eye < self.ear_thresh:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1  # blink count
                        COUNTER = 0

            eye_closed_flag = 1 if (eye is not None and eye < self.ear_thresh) else 0
            eye_closed_win.append(eye_closed_flag)

            if mouth is not None:
                if mouth > self.mar_thresh:
                    mCOUNTER += 1
                else:
                    if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        mTOTAL += 1  # yawn count
                        mCOUNTER = 0

            mouth_open_flag = 1 if (mouth is not None and mouth > self.mar_thresh) else 0
            mouth_open_win.append(mouth_open_flag)

            # ====== perclos compute on sliding window ======
            fatigue_alarm = None
            perclos = None
            if len(eye_closed_win) == PERCL0S_WINDOW and len(mouth_open_win) == PERCL0S_WINDOW:
                perclos = (sum(eye_closed_win) / PERCL0S_WINDOW) + (sum(mouth_open_win) / PERCL0S_WINDOW) * 0.2
                perclos = float(perclos)
                fatigue_alarm = perclos > self.perclos_thresh
                perclos_log_counter += 1

                # 滑动窗口逐帧计算，日志按状态变化或固定间隔输出，避免刷屏
                if fatigue_alarm != last_fatigue_alarm or (perclos_log_counter % 30 == 1):
                    self.log_signal.emit(f"最近{PERCL0S_WINDOW}帧(滑动窗口) Perclos得分为 {perclos:.3f}")
                    if fatigue_alarm:
                        self.log_signal.emit("当前处于疲劳状态")
                    else:
                        self.log_signal.emit("当前处于清醒状态")
                    self.log_signal.emit("")
                last_fatigue_alarm = fatigue_alarm

            st = {
                "fps": fps,
                "eye": float(eye) if eye is not None else None,
                "mouth": float(mouth) if mouth is not None else None,
                "blink_total": int(TOTAL),
                "yawn_total": int(mTOTAL),
                "perclos": perclos,
                "fatigue_update": fatigue_alarm,
                "phone": bool(phone_hit),
                "smoke": bool(smoke_hit),
                "drink": bool(drink_hit),
                "distract_any": distract_any,
            }

            self.status_signal.emit(st)
            self.frame_signal.emit(frame)

        cap.release()
        self.log_signal.emit("Camera stopped.")