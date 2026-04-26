# Fatigue-and-distraction-driving-detection-based-on-YOLOv8-and-PFLD 权重文件
This is a project on fatigue driving and distracted behavior detection, using YOLOv8 and PFLD models, which can be deployed on the domestically produced Huawei Atlas 200 edge device.这个分支提供了权重文件，*.pt为pytorch的权重文件，onnx为开源架构格式，om为专用于昇腾环境的模型架构

best.pt：YOLOv8 训练得到的 PyTorch 权重文件（原始精度）。

best.om：YOLOv8 转换后的离线模型（.om 格式），适配 Atlas 200 推理。

pfld_106.onnx：PFLD 关键点检测模型的 ONNX 导出文件（106 个关键点）。

pfld_106.om：PFLD 模型转换后的 Atlas 200 离线模型。


<img width="702" height="204" alt="image" src="https://github.com/user-attachments/assets/ada5dac1-aa66-4336-ad14-06a0c5b41ca7" />

