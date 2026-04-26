# src/app.py
import cv2
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QMainWindow, QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QLabel, QDoubleSpinBox, QSpinBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from ui_mainwindow import Ui_MainWindow
from camera_worker import CameraWorker
from config import CAMERA_ID, EAR_THRESH, MAR_THRESH, PERCL0S_THRESH, PFLD_ONNX_PATH
from pfld_onnx import Pfld106
from multitask_pipeline import frametest
from test_baseline_ear_mar import build_yolo_backend, summarize


class DriverMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label.setText("欢迎使用驾驶员状态监测系统\n\n请点击右上角菜单open：\n\n启动驾驶状态监测 或 采集EAR/MAR/PERCLOS基线")
        # 可选：居中 + 样式（如果你没在ui文件里加样式，也可以在这里加）
        from PySide6.QtCore import Qt
        self.ui.label.setAlignment(Qt.AlignCenter)
        self.ui.label.setWordWrap(True)
        self.ui.label.setStyleSheet('''
            QLabel#label {
                background-color: rgb(245, 248, 255);
                border: 2px solid rgb(210, 220, 240);
                border-radius: 12px;
                color: rgb(30, 45, 70);
                padding: 18px;
            }
''')
        # ------- UI init text -------
        #self.ui.label.setText("Camera not started")
        self.ui.label_2.setText("FPS:")
        self.ui.label_10.setText("清醒")          # 疲劳/清醒
        self.ui.label_3.setText("眨眼次数：0")
        self.ui.label_4.setText("哈欠次数：0")
        self.ui.label_5.setText("提示：")
        self.ui.label_9.setText("")              # 分心提示

        self.ui.label_6.setText("手机")
        self.ui.label_7.setText("抽烟")
        self.ui.label_8.setText("喝水")

        # ------- theme styles -------
        self._normal_style = ""  # 恢复为Qt默认
        # 红色预警：centralwidget背景变浅红 + 右侧文字更醒目
        self._alert_style = """
        QWidget#centralwidget {
            background-color: rgb(255, 0, 0);
        }
        QLabel {
            font-weight: 600;
        }
        """

        # ------- state latch (avoid flicker) -------
        # fatigue_update 不是每帧都有，所以需要锁存最近一次疲劳结论
        self._fatigue_latched = False
        self._last_ear_thresh = float(EAR_THRESH)
        self._last_mar_thresh = float(MAR_THRESH)
        self._last_perclos_thresh = float(PERCL0S_THRESH)
        self._baseline_running = False

        self.worker = None
        self.ui.actionOpen_camera.triggered.connect(self.on_open_camera)
        self.ui.actionBaseline.triggered.connect(self.on_run_baseline)
        self.ui.printf("Ready. Click menu to start monitoring or run baseline calibration.")

    def _apply_light_dialog_style(self, dlg: QDialog):
        dlg.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
                color: #111111;
            }
            QLabel {
                color: #111111;
                background: transparent;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #FFFFFF;
                color: #111111;
                border: 1px solid #CFCFCF;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1px solid #CFCFCF;
                border-radius: 4px;
                padding: 4px 12px;
                min-width: 72px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
            }
            """
        )

    def _ask_thresholds(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("阈值设置")
        self._apply_light_dialog_style(dlg)
        layout = QVBoxLayout(dlg)

        tip = QLabel("请输入本次监测使用的阈值：\nEAR 越小越容易判定闭眼，MAR 越小越容易判定张嘴，PERCLOS 越小越容易触发疲劳。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        form = QFormLayout()
        ear_spin = QDoubleSpinBox(dlg)
        ear_spin.setDecimals(3)
        ear_spin.setRange(0.01, 1.00)
        ear_spin.setSingleStep(0.005)
        ear_spin.setValue(self._last_ear_thresh)
        form.addRow("EAR 阈值（闭眼判定，EAR < 阈值）", ear_spin)

        mar_spin = QDoubleSpinBox(dlg)
        mar_spin.setDecimals(3)
        mar_spin.setRange(0.01, 2.00)
        mar_spin.setSingleStep(0.01)
        mar_spin.setValue(self._last_mar_thresh)
        form.addRow("MAR 阈值（张嘴判定，MAR > 阈值）", mar_spin)

        perclos_spin = QDoubleSpinBox(dlg)
        perclos_spin.setDecimals(3)
        perclos_spin.setRange(0.01, 1.20)
        perclos_spin.setSingleStep(0.01)
        perclos_spin.setValue(self._last_perclos_thresh)
        form.addRow("PERCLOS 阈值（疲劳判定，PERCLOS > 阈值）", perclos_spin)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None

        self._last_ear_thresh = float(ear_spin.value())
        self._last_mar_thresh = float(mar_spin.value())
        self._last_perclos_thresh = float(perclos_spin.value())
        return self._last_ear_thresh, self._last_mar_thresh, self._last_perclos_thresh

    def _ask_baseline_params(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("基线采集参数")
        self._apply_light_dialog_style(dlg)
        layout = QVBoxLayout(dlg)

        tip = QLabel("请输入基线采集参数：\n有效帧数越大，建议阈值越稳定；预热帧用于让曝光和检测先稳定。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        form = QFormLayout()
        frames_spin = QSpinBox(dlg)
        frames_spin.setRange(100, 2000)
        frames_spin.setSingleStep(10)
        frames_spin.setValue(300)
        form.addRow("采集有效帧数", frames_spin)

        warmup_spin = QSpinBox(dlg)
        warmup_spin.setRange(0, 300)
        warmup_spin.setSingleStep(5)
        warmup_spin.setValue(30)
        form.addRow("预热帧数", warmup_spin)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None
        return int(frames_spin.value()), int(warmup_spin.value())

    def on_open_camera(self):
        if self._baseline_running:
            self.ui.printf("Baseline calibration is running. Please wait.")
            return

        if self.worker is not None and self.worker.isRunning():
            self.ui.printf("Monitoring is already running.")
            return

        thresholds = self._ask_thresholds()
        if thresholds is None:
            self.ui.printf("Threshold input canceled.")
            return

        ear_thresh, mar_thresh, perclos_thresh = thresholds

        self.worker = CameraWorker(
            camera_id=CAMERA_ID,
            ear_thresh=ear_thresh,
            mar_thresh=mar_thresh,
            perclos_thresh=perclos_thresh,
        )
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.status_signal.connect(self.update_status)
        self.worker.log_signal.connect(self.ui.printf)
        self.worker.start()

        self.ui.printf(
            f"Starting monitor with thresholds: EAR<{ear_thresh:.3f}, "
            f"MAR>{mar_thresh:.3f}, PERCLOS>{perclos_thresh:.3f}"
        )

    def on_run_baseline(self):
        if self.worker is not None and self.worker.isRunning():
            self.ui.printf("Please stop monitoring before baseline calibration.")
            return
        if self._baseline_running:
            self.ui.printf("Baseline calibration is already running.")
            return

        baseline_params = self._ask_baseline_params()
        if baseline_params is None:
            return
        frames, warmup = baseline_params

        self._baseline_running = True
        self.ui.printf("开始采集基线，请保持睁眼、闭嘴、正视前方...")
        QApplication.processEvents()

        cap = None
        try:
            yolo_backend = build_yolo_backend()
            pfld_model = Pfld106(PFLD_ONNX_PATH)
            cap = cv2.VideoCapture(CAMERA_ID)
            if not cap.isOpened():
                self.ui.printf(f"Cannot open camera id={CAMERA_ID}")
                return

            ears = []
            mars = []
            perclos_series = []
            eye_closed_win = deque(maxlen=150)
            mouth_open_win = deque(maxlen=150)

            frame_idx = 0
            valid_idx = 0

            while valid_idx < frames:
                QApplication.processEvents()
                ok, frame = cap.read()
                if not ok:
                    self.ui.printf("Camera read failed during baseline calibration.")
                    break

                frame_idx += 1
                if frame_idx <= warmup:
                    continue

                (lab, eye, mouth, dets, kps_frame), _ = frametest(yolo_backend, pfld_model, frame)
                if eye is None or mouth is None:
                    continue

                ears.append(float(eye))
                mars.append(float(mouth))
                valid_idx += 1

                eye_closed_win.append(1 if eye < self._last_ear_thresh else 0)
                mouth_open_win.append(1 if mouth > self._last_mar_thresh else 0)
                if len(eye_closed_win) == 150:
                    p = (sum(eye_closed_win) / 150.0) + (sum(mouth_open_win) / 150.0) * 0.2
                    perclos_series.append(float(p))

                if valid_idx % 50 == 0:
                    self.ui.printf(f"Baseline progress: {valid_idx}/{frames}")

            self.ui.printf(f"Baseline done. valid={len(ears)} frames")
            if len(ears) < max(30, frames // 3):
                self.ui.printf("有效样本偏少，请调整姿态和光照后重试。")
                return

            ear_stats = summarize(ears, "EAR")
            mar_stats = summarize(mars, "MAR")
            perclos_stats = summarize(perclos_series, "PERCLOS") if len(perclos_series) > 0 else None

            ear_suggest = max(0.05, ear_stats["p10"] * 0.90)
            mar_suggest = mar_stats["p90"] * 1.10
            if perclos_stats is not None:
                perclos_suggest = max(0.05, perclos_stats["p90"] * 1.10)
            else:
                perclos_suggest = self._last_perclos_thresh

            self._last_ear_thresh = float(ear_suggest)
            self._last_mar_thresh = float(mar_suggest)
            self._last_perclos_thresh = float(perclos_suggest)

            self.ui.printf("=== 基线统计 ===")
            self.ui.printf(
                f"EAR mean={ear_stats['mean']:.4f}, p10={ear_stats['p10']:.4f}, p90={ear_stats['p90']:.4f}"
            )
            self.ui.printf(
                f"MAR mean={mar_stats['mean']:.4f}, p10={mar_stats['p10']:.4f}, p90={mar_stats['p90']:.4f}"
            )
            if perclos_stats is not None:
                self.ui.printf(
                    f"PERCLOS mean={perclos_stats['mean']:.4f}, p10={perclos_stats['p10']:.4f}, p90={perclos_stats['p90']:.4f}"
                )
            self.ui.printf(
                f"建议阈值: EAR<{ear_suggest:.3f}, MAR>{mar_suggest:.3f}, PERCLOS>{perclos_suggest:.3f}"
            )

        except Exception as e:
            self.ui.printf(f"Baseline calibration failed: {e}")
        finally:
            self._baseline_running = False
            if cap is not None:
                cap.release()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        super().closeEvent(event)

    def update_frame(self, frame_bgr: np.ndarray):
        show = cv2.resize(frame_bgr, (720, 480))
        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(qimg))

    def _set_alert_theme(self, is_alert: bool):
        # 只在状态变化时切换，避免频繁 setStyleSheet
        current = self.centralWidget().styleSheet()
        target = self._alert_style if is_alert else self._normal_style
        if current != target:
            self.centralWidget().setStyleSheet(target)

    def update_status(self, st: dict):
        # FPS
        fps = st.get("fps", 0.0)
        self.ui.label_2.setText(f"FPS: {fps:.1f}")

        # blink/yawn totals
        self.ui.label_3.setText("眨眼次数：" + str(st.get("blink_total", 0)))
        self.ui.label_4.setText("哈欠次数：" + str(st.get("yawn_total", 0)))

        # distraction labels
        phone = bool(st.get("phone", False))
        smoke = bool(st.get("smoke", False))
        drink = bool(st.get("drink", False))
        distract_any = bool(st.get("distract_any", False))

        self.ui.label_6.setText("<font color=red>正在用手机</font>" if phone else "手机")
        self.ui.label_7.setText("<font color=red>正在抽烟</font>" if smoke else "抽烟")
        self.ui.label_8.setText("<font color=red>正在喝水</font>" if drink else "喝水")
        self.ui.label_9.setText("<font color=red>请不要分心</font>" if distract_any else "")

        # fatigue latch update (only when worker computed perclos window)
        fatigue_update = st.get("fatigue_update", None)  # None / True / False
        if fatigue_update is True:
            self._fatigue_latched = True
        elif fatigue_update is False:
            self._fatigue_latched = False

        # fatigue label (based on latched state)
        if self._fatigue_latched:
            self.ui.label_10.setText("<font color=red>疲劳！！！</font>")
        else:
            self.ui.label_10.setText("清醒")

        # ---- ALERT THEME: red UI when distraction OR fatigue ----
        is_alert = distract_any or self._fatigue_latched
        self._set_alert_theme(is_alert)