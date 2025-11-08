from ultralytics import YOLO

# ----------------------------------------------------
# 载入预训练模型（可以是 yolov8n.pt / yolov8s.pt / yolov8m.pt）
# yolov8n.pt 最轻，适合先跑通流程
# ----------------------------------------------------
model = YOLO("yolo11s.pt")

# ----------------------------------------------------
# 开始训练
# ----------------------------------------------------
results = model.train(
    data="/root/autodl-tmp/dataset/data.yaml",  # 保持不变
    epochs=100,      # 训练轮数
    imgsz=320,       # 可以 320 / 384 / 512，看显存
    batch=16,        # 视显存大小适当调
    workers=4,       # 线程数
    device=0         # GPU
)

# 训练结果（模型权重）默认保存在 runs/detect/train*/weights/best.pt
