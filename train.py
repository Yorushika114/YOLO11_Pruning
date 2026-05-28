# 从ultralytics库中导入YOLO类
from ultralytics import YOLO
import torch
import torch.nn as nn


# 加载预训练模型
# model = YOLO("YOLO11n.pt")

# 用一个自定义的配置文件在数据集上训练模型，epochs(周期)，imgsz(图片像素大小)，batch(批次，根据你设备的性能而定，会影响模型训练速度和训练效果。)
# 说白了就类似于GPU或者CPU（还有最近华为推出的算力芯片，是以NPU形式出现）根据物理界限，每一批要出去的人数。
# results = model.train(data='./data.yaml',epochs=100,imgsz=640,batch=16)

def train_model():
    model = YOLO('/gemini/code/Torch-Pruning-master/examples/yolov8/yolo11n.pt')  # 加载模型
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)  # 无形状错误则说明修改生效
    print("没有任何错误")
    # # 强制BatchNorm层兼容小尺寸输入
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eps = 1e-3  # 增大eps，避免数值不稳定
    #         m.momentum = 0.03  # 减小动量，适配小批量

    model.train(data=r"/gemini/code/datasets/coco128.yaml", epochs=400, batch=32, workers=16,
                imgsz=640, save=True, resume=False)  # workers > 0 时要注意


if __name__ == '__main__':
    train_model()
    # model = YOLO(r"C:\Users\23223\Desktop\识别杯子.pt")
    # results = model.val()
    # results = model.predict(source=r"C:\Users\23223\Desktop\新建文件夹",
    #                         conf=0.25,
    #                         save=True)

# 评估模型在验证集上的性能weish