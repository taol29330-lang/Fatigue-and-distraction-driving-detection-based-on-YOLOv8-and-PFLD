# Fatigue-and-distraction-driving-detection-based-on-YOLOv8-and-PFLD
This is a project on fatigue driving and distracted behavior detection, using YOLOv8 and PFLD models, which can be deployed on the domestically produced Huawei Atlas 200 edge device.


# YOLOv8 + PFLD 驾驶员状态监测系统

一个基于摄像头的驾驶员状态监测项目，支持以下能力：

- 疲劳检测：基于 PFLD 106 关键点计算 EAR、MAR、PERCLOS
- 分心行为检测：手机、抽烟、喝水
- 可视化 UI：实时画面、告警状态、日志输出
- 基线采集：采集个人 EAR/MAR/PERCLOS 基线并给出阈值建议

项目当前以 PySide6 桌面应用为主入口。

## 1. 功能概览
<img width="721" height="374" alt="image" src="https://github.com/user-attachments/assets/a4bb3509-9748-4b7f-81b3-231d9b8b3fcf" />

### 1.1 检测能力

- 人脸与分心目标检测：YOLOv8
- 面部关键点回归：PFLD ONNX（106 点）
- 疲劳指标：
  - EAR（Eye Aspect Ratio）
  - MAR（Mouth Aspect Ratio）
  - PERCLOS（滑动窗口）

### 1.2 疲劳判定逻辑
<img width="702" height="225" alt="image" src="https://github.com/user-attachments/assets/df1c91da-ab3b-4c7e-ae14-73fe1b8b53c1" />

- 闭眼判定：EAR < EAR_THRESH
- 张嘴判定：MAR > MAR_THRESH
- 窗口大小：150 帧
- PERCLOS 计算：

```text
PERCLOS = (闭眼帧占比) + 0.2 * (张嘴帧占比)
```

- 疲劳告警：PERCLOS > PERCL0S_THRESH

### 1.3 分心判定逻辑

- 关注类别：Phone / Smoke / Drink
- 命中后在 UI 显示对应分心提示
- 分心与疲劳任一触发时，界面进入红色告警主题

## 2. 项目结构

```text
.
├─ app.py                    # 主窗口逻辑（UI交互、阈值输入、基线采集入口）
├─ main.py                   # 应用启动入口
├─ camera_worker.py          # 摄像头线程，整合检测、统计、状态输出
├─ multitask_pipeline.py     # 单帧多任务推理流程（YOLO + PFLD）
├─ fatigue_metrics.py        # EAR/MAR 指标计算
├─ head_pose.py              # 头姿估计（solvePnP），当前主流程未强依赖
├─ yolo_ultralytics.py       # YOLOv8 推理封装
├─ pfld_onnx.py              # PFLD ONNX 推理封装
├─ test_baseline_ear_mar.py  # 命令行基线采集脚本
├─ ui_mainwindow.py          # PySide6 UI 定义
├─ config.py                 # 全局配置（路径、阈值、相机、后端等）
├─ utils.py                  # 通用工具函数
└─ weights/
   ├─ best.pt
   ├─ best.om
   ├─ pfld_106.onnx
   └─ pfld_106.om
```

## 3. 环境要求

推荐环境：

- Windows 10/11
- Python 3.9 - 3.11
- 摄像头设备可用

核心依赖：

- numpy
- opencv-python
- onnxruntime
- ultralytics
- PySide6

## 4. 安装步骤

### 4.1 创建虚拟环境（可选但推荐）

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4.2 安装依赖

```bash
pip install --upgrade pip
pip install numpy opencv-python onnxruntime ultralytics PySide6
```

### 4.3 检查模型文件

确认以下文件存在：

- weights/best.pt
- weights/pfld_106.onnx

如果你要切换 MindX OM 后端，请确认对应 .om 文件和后端封装可用。

## 5. 快速开始

### 5.1 启动图形界面

```bash
python main.py
```
<img width="2065" height="1262" alt="image" src="https://github.com/user-attachments/assets/3a5489df-9407-4350-8bb9-fc1307160722" />

启动后在菜单中可执行：

- open -> 启动驾驶状态监测
- open -> 采集 EAR/MAR/PERCLOS 基线

### 5.2 命令行采集基线

```bash
python test_baseline_ear_mar.py --show
```

常用参数：

- --camera: 摄像头 ID（默认取 config.py 中 CAMERA_ID）
- --frames: 有效采样帧数（默认 300）
- --warmup: 预热帧数（默认 30）
- --show: 显示采样窗口

## 6. 关键配置说明（config.py）

常改参数：

- YOLO_BACKEND: ultralytics 或 mindx_om
- YOLO_PT_PATH / YOLO_OM_PATH: YOLO 模型路径
- PFLD_ONNX_PATH: PFLD ONNX 模型路径
- CAMERA_ID: 摄像头编号
- YOLO_CONF / YOLO_IOU: 检测阈值
- DRIVER_SELECT: 驾驶员人脸选择策略（max_x2 / largest_area）
- EAR_THRESH / MAR_THRESH / PERCL0S_THRESH: 疲劳相关阈值

建议流程：

1. 先执行基线采集（或在 UI 中进行基线采集）。
2. 根据建议值更新阈值。
3. 上车实测后小范围微调。

## 7. 数据流与架构

```text
摄像头帧
  -> YOLO 检测（Face/Phone/Smoke/Drink）
  -> 选择驾驶员人脸框
  -> 裁剪 + PFLD 关键点
  -> 计算 EAR / MAR
  -> 滑动窗口计算 PERCLOS
  -> 状态输出到 UI（清醒/疲劳 + 分心提示 + 统计）
```

核心线程模型：

- 主线程：UI 展示、菜单操作、阈值输入
- 工作线程：camera_worker 持续采集与推理

## 8. 常见问题

### 8.1 摄像头打不开

- 检查 CAMERA_ID 是否正确（0/1/2...）
- 关闭占用摄像头的软件
- 确认系统摄像头权限已开启

### 8.2 帧率低

- 降低摄像头分辨率
- 降低 YOLO 输入尺寸或降低检测频率（需改代码）
- 优先使用性能更好的推理设备/后端

### 8.3 疲劳误报或漏报

- 重新采集个人基线，避免直接套用默认阈值
- 检查光照、摄像头角度、面部遮挡
- 微调 EAR/MAR/PERCLOS 阈值

### 8.4 模型加载失败

- 检查权重路径是否与 config.py 一致
- 确认 onnxruntime、ultralytics 版本与 Python 版本兼容


### 8.5 边缘配置
- 如需在边缘配置，可以修改模型推理部分，在华为Atlas边缘设备部署推理速度约为25FPS，基本满足实时监测要求；
- 权重文件已上传，参考“weights/*.om"
- 修改方法可参考https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Getting%20Started%20with%20Application%20Development/gswaad/gswaad_0002.html
- 本项目未上传边缘设备部署的代码，如有需要可以QQ联系：2052329330

## 9. 开发建议

- 增加 requirements.txt，固定依赖版本，提升可复现性
- 为不同设备（白天/夜间）建立多组阈值配置
- 在日志中记录告警片段，便于后处理分析
- 补充单元测试与回归测试脚本

## 10. 免责声明

本项目用于学习与研究。实际车载部署需进行严格的安全验证、性能评估与法规合规审查，不可直接作为商业安全产品替代方案。


参考资料如下：

https://www.hiascend.com/developer/blog/details/0215172312608801248

庞 夏 君 . 基 于 深 度 学 习 的 驾 驶 员 疲 劳 和 分 心 状 态 识 别 算 法 研 究 [D]. 广 西 民 族 大 学,2024.DOI:10.27035/d.cnki.ggxmc.2024.000578.

PFLD权重文件：https://gitee.com/link?target=https%3A%2F%2Fmindx.sdk.obs.cn-north-4.myhuaweicloud.com%2Fmindxsdk-referenceapps%2520%2Fcontrib%2FFatigueDrivingRecognition%2Fmodel.zip

特别感谢以上作者的公开资料，如有侵权可联系删除。

