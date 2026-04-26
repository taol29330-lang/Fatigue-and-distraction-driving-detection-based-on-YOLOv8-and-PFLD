# src/yolo_ultralytics.py
import numpy as np
from ultralytics import YOLO

class YoloUltralytics:
    def __init__(self, model_path: str, conf=0.6):
        self.model = YOLO(model_path, task="detect")
        self.conf = conf

    def infer(self, bgr):
        """
        return list[dict]:
          { 'cls': int, 'name': str, 'conf': float, 'xyxy': np.ndarray shape(4,) }
        """
        r = self.model(bgr, conf=self.conf, verbose=False)[0]
        if r.boxes is None or r.boxes.xyxy is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(np.int32)

        out = []
        for i in range(len(xyxy)):
            out.append({"cls": int(cls[i]),
                        "name": self.model.names[int(cls[i])],
                        "conf": float(conf[i]),
                        "xyxy": xyxy[i]})
        return out