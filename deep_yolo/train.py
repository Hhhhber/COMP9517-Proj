from ultralytics import YOLO

# ----------------------------------------------------
# ----------------------------------------------------
model = YOLO("yolo11s.pt")

# ----------------------------------------------------
# 开始训练
# ----------------------------------------------------
results = model.train( 
 data="/root/AgroPest-12/data.yaml" # 修改：更改为项目统一数据根 
 epochs=100, 
 imgsz=320, 
 batch=16, 
 workers=4, 
 device=0, 
 project="./outputs/deep_learning", # 新增：统一输出根 
 name="train" # 新增：run 名称 
)
# 训练结果（模型权重）默认保存在 runs/detect/train*/weights/best.pt
