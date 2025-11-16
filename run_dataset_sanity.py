#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random, argparse
from PIL import Image, ImageDraw, ImageFont

# 尝试从多处导入你们的 Dataset，按项目情况改动优先级
COCO_DS = None
try:
    from data.dataset import COCODetection as COCO_DS
except Exception:
    try:
        from dataset import COCODetection as COCO_DS
    except Exception:
        COCO_DS = None

def to_pil(img):
    if isinstance(img, Image.Image):
        return img
    try:
        from torchvision.transforms.functional import to_pil_image
        return to_pil_image(img)
    except Exception:
        # 兜底：若是numpy
        import numpy as np
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        raise TypeError("无法将图像转换为PIL，请在Dataset中返回PIL或tensor/ndarray")

def draw_image(img, boxes, labels, out_path):
    im = to_pil(img).convert("RGB")
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.load_default()
    except: font = None
    for b,lab in zip(boxes,labels):
        x,y,w,h = map(float, b)
        dr.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        dr.text((x,y), str(int(lab)), fill=(255,0,0), font=font)
    im.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)       # 例如 data/val.json
    ap.add_argument("--image_root", required=True) # 图片根目录
    ap.add_argument("--out", default="preview")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()

    if COCO_DS is None:
        raise SystemExit("未找到 COCODetection，请确认导入路径（data/dataset.py 或 dataset.py）")

    os.makedirs(a.out, exist_ok=True)
    ds = COCO_DS(a.json, a.image_root, transforms=None)
    total = len(ds)
    if total == 0:
        raise SystemExit("Dataset 为空，请检查 json 与 image_root 是否匹配。")
    idxs = random.sample(range(total), min(a.n, total))
    for i,k in enumerate(idxs):
        img, target = ds[k]
        boxes = target.get("boxes", [])
        labels = target.get("labels", [])
        out_path = os.path.join(a.out, f"vis_{i:02d}.jpg")
        draw_image(img, boxes, labels, out_path)
    print(f"[OK] 已保存 {len(idxs)} 张可视化到 {a.out}/")

if __name__ == "__main__":
    main()