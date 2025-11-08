# Deep Learning Module – YOLO Object Detection

This folder contains the **Deep Learning** part of our COMP9517 group project.  
We trained and evaluated **YOLOv11s** (Ultralytics) for 12-class pest detection.

---

## Environment
- Python 3.10
- PyTorch 2.1.2 + CUDA 11.8
- Ultralytics 8.3.225
- GPU: NVIDIA RTX 4090 (AutoDL)

---

## Dataset
Configured via `data.yaml`:
```yaml
train: /root/autodl-tmp/dataset/train/images
val:   /root/autodl-tmp/dataset/val/images
test:  /root/autodl-tmp/dataset/test/images
nc: 12
names: [Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs,
        Grasshoppers, Moths, Slugs, Snails, Wasps, Weevils]
```

---

## Training
Script: `train.py`
```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')   # YOLOv11 small
results = model.train(
    data='/root/autodl-tmp/dataset/data.yaml',
    epochs=100,
    imgsz=320,
    batch=16,
    workers=4,
    device=0
)
```
Outputs (weights, logs, curves) are saved to:
```
/root/autodl-tmp/yolo/runs/detect/train5/
```

---

## Testing & Evaluation
Script: `test.py`
```bash
python test.py
```
**Final metrics (test set):**
- Precision: **0.8666**
- Recall: **0.7405**
- F1-score: **0.7986**
- Accuracy (from confusion matrix): **0.6803**
- mAP@0.5: **0.8024**
- mAP@0.5:0.95: **0.4992**

Evaluation artifacts are saved under:
```
/root/autodl-tmp/yolo/runs/detect/val*/
```

---

## Visualization (Predictions)
Generate detection examples on test images:
```bash
yolo predict   model=/root/autodl-tmp/yolo/runs/detect/train5/weights/best.pt   source=/root/autodl-tmp/dataset/test/images   save=True imgsz=640 conf=0.25 device=0
```
Results (annotated images) will be in:
```
/root/autodl-tmp/yolo/runs/detect/predict*/
```

Use several images from this folder in the report/demo.

---

## Notes
- `best.pt` is under `runs/detect/train5/weights/`.
- Curves: `results.png`, `PR_curve.png`, `confusion_matrix.png` are in the same run folder.
- This README belongs in `/root/autodl-tmp/yolo/`.

