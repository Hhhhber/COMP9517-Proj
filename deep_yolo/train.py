from ultralytics import YOLO

# ----------------------------------------------------
# ----------------------------------------------------
model = YOLO("yolo11s.pt")

# ----------------------------------------------------
# 开始训练
# ----------------------------------------------------
results = model.train( 
 data="./data/AgroPest-12/data.yaml",
 epochs=1,
 imgsz=64,
 batch=1,
 workers=1,
 device='cpu',
 project="./outputs/deep_yolo",
 name="train" #
)
# 训练结果（模型权重）默认保存在 ./outputs/deep_yolo/train/weights/best.pt
