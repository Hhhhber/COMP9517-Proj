# Faster R-CNN (f_cnn)

This folder contains the Faster R-CNN implementation for our project. It includes training, evaluation, visualization, and unified-JSON export for integration.

## Environment
pip install torch torchvision torchaudio
pip install pycocotools tqdm matplotlib pillow numpy opencv-python albumentations scikit-learn
# or
pip install -r requirements.txt

## Train
python -m f_cnn.train_frcnn \
  --data_root "$DATA" \
  --epochs 50 --batch_size 4 --imgsz 640 \
  --device cuda

## Evaluate (mAP / P / R)
# Valid (baseline 640)
python -m f_cnn.eval_frcnn \
  --data_root "$DATA" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --split valid --imgsz 640 --conf 0.6 --max_det 300 \
  --out "/root/autodl-tmp/f_cnn/outputs3/eval_final.json"

# (Optional) Valid @800
python -m f_cnn.eval_frcnn \
  --data_root "$DATA" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --split valid --imgsz 800 --conf 0.6 --max_det 300 \
  --out "/root/autodl-tmp/f_cnn/outputs3/eval_valid_img800.json"

# Train / Test (optional)
python -m f_cnn.eval_frcnn --data_root "$DATA" --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" --split train --conf 0.6 --out "/root/autodl-tmp/f_cnn/outputs3/eval_train.json"
python -m f_cnn.eval_frcnn --data_root "$DATA" --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" --split test  --conf 0.6 --out "/root/autodl-tmp/f_cnn/outputs3/eval_test.json"

## Visualization (samples)
python -m f_cnn.infer_frcnn \
  --data_root "$DATA" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --split valid --imgsz 640 --save_vis 1 --vis_limit 30 \
  --out_dir "/root/autodl-tmp/f_cnn/vis_valid"

## Export unified JSON (for integration)
# Train
PYTHONPATH=. python tools/export_from_dir.py \
  --images_dir "$DATA/train/images" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --conf 0.6 \
  --out "/root/autodl-tmp/f_cnn/outputs5/preds_trad_train.json"

# Valid
PYTHONPATH=. python tools/export_from_dir.py \
  --images_dir "$DATA/valid/images" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --conf 0.6 \
  --out "/root/autodl-tmp/f_cnn/outputs5/preds_trad_valid.json"

# Test
PYTHONPATH=. python tools/export_from_dir.py \
  --images_dir "$DATA/test/images" \
  --weights "/root/autodl-tmp/f_cnn/runs_res50_clean/best.pth" \
  --conf 0.6 \
  --out "/root/autodl-tmp/f_cnn/outputs5/preds_trad_test.json"

## Notes
- Inference threshold fixed in infer_frcnn.py: score_thresh=0.6, nms_thresh=0.5, detections_per_img=300.
- eval_frcnn.py saves evaluation JSON (mAP/P/R and per-class AP).
- Unified JSON files (preds_trad_*.json) are used for integration with the traditional (SVM) pipeline.

