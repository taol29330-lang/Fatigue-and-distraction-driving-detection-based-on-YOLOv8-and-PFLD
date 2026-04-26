'''
import numpy as np
import onnxruntime as ort

class Pfld106:
    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        # 常见PFLD导出：只有一个输出；也有多个输出。这里优先取第一个输出做关键点
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def infer(self, face_bgr_112) -> np.ndarray:
        outs = self.sess.run(self.output_names, {self.input_name: inp})
        """
        face_bgr_112: (112,112,3) BGR uint8/float
        return keypoints: (106,2) in 112x112 coordinate space (float)
        """
        img = face_bgr_112.astype(np.float32)
        img = img / 255.0
        # BGR -> RGB (很多PFLD训练用RGB；若你模型是BGR训练，删掉这一行)
        img = img[:, :, ::-1]
        inp = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW

        outs = self.sess.run(self.output_names, {self.input_name: inp})
        # 关键点通常在第0个输出；如果你发现形状不对，再切换到 outs[1]
        kps = outs[0]
        kps = np.array(kps).reshape(-1, 2) * 112.0
        if kps.shape[0] != 106:
            # 兼容：有些导出把关键点放在第二个输出
            kps2 = np.array(outs[-1]).reshape(-1, 2) * 112.0
            if kps2.shape[0] == 106:
                kps = kps2
        return kps.astype(np.float32)
'''
import numpy as np
import onnxruntime as ort

class Pfld106:
    def __init__(self, onnx_path: str, providers=None,
                 debug: bool = False, debug_print_times: int = 3,
                 logger=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

        self.debug = bool(debug)
        self._debug_left = int(debug_print_times)
        self.logger = logger if logger is not None else print

    def _log(self, msg: str):
        try:
            self.logger(msg)
        except Exception:
            pass

    # [NEW] 自动从多输出里选择“像关键点”的那个输出（优先106*2或212）
    def _pick_keypoints_output(self, outs):
        candidates = []
        for i, o in enumerate(outs):
            arr = np.asarray(o).squeeze()
            # 常见： (1,212) / (212,) / (1,106,2) / (106,2)
            if arr.ndim == 1 and arr.size in (212,):
                candidates.append((i, arr))
            elif arr.ndim == 2 and arr.shape == (106, 2):
                candidates.append((i, arr))
            elif arr.ndim == 3 and arr.shape[-2:] == (106, 2):
                candidates.append((i, arr[0]))
        if candidates:
            # 选第一个命中的即可；一般就是最后一个输出
            return candidates[0][0], candidates[0][1]

        # fallback：按你现在观察，outs[-1] 才对
        arr = np.asarray(outs[-1]).squeeze()
        return len(outs) - 1, arr

    # [NEW] 自动判断是否需要缩放到112像素
    def _maybe_scale_to_112(self, kps):
        kps = kps.astype(np.float32)
        mn, mx = float(kps.min()), float(kps.max())
        # 情况1：输出在[0,1]左右 -> 乘112
        if mx <= 2.0 and mn >= -0.5:
            return kps * 112.0
        # 情况2：输出在[-1,1] -> 映射到[0,112]
        if mn >= -1.5 and mx <= 1.5:
            return (kps + 1.0) * 0.5 * 112.0
        # 情况3：看起来已经是像素坐标 -> 不缩放
        return kps

    def infer(self, face_bgr_112) -> np.ndarray:
        img = face_bgr_112.astype(np.float32) / 255.0
        img = img[:, :, ::-1]  # BGR->RGB（如你的模型是BGR训练可删）
        inp = np.transpose(img, (2, 0, 1))[None, ...]  # 1,3,112,112

        outs = self.sess.run(self.output_names, {self.input_name: inp})

        # [NEW] 选真正关键点输出
        out_idx, out_arr = self._pick_keypoints_output(outs)

        # [NEW] reshape成(106,2)
        out_arr = np.asarray(out_arr).squeeze()
        if out_arr.ndim == 1 and out_arr.size == 212:
            kps = out_arr.reshape(106, 2)
        elif out_arr.ndim == 2 and out_arr.shape == (106, 2):
            kps = out_arr
        else:
            kps = out_arr.reshape(-1, 2)

        # [NEW] 缩放到112坐标（仅在需要时）
        kps = self._maybe_scale_to_112(kps)

        # [NEW] debug输出（只打印前N次）
        if self.debug and self._debug_left > 0:
            self._debug_left -= 1
            self._log(f"PFLD pick outs[{out_idx}] as keypoints, raw_shape={np.asarray(out_arr).shape}")
            self._log(f"kps shape={kps.shape}, min={float(kps.min()):.3f}, max={float(kps.max()):.3f}")

        # [NEW] 强制只取106点（防御性）
        if kps.shape[0] != 106:
            self._log(f"[WARN] keypoints count != 106, got {kps.shape[0]}")
        kps = kps[:106, :]

        return kps.astype(np.float32)