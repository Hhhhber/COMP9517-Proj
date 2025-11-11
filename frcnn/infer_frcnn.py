# f_cnn/infer_frcnn.py
import os, json, time, argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from f_cnn.data import YoloDetectDataset, get_transforms, collate_fn

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # 无 tqdm 也能跑

def load_model(ckpt_path: str, num_classes: int, device: torch.device):
    """加载权重并构建相同 head 的 Faster R-CNN。"""
    state = torch.load(ckpt_path, map_location="cpu")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    model.load_state_dict(state["model"])
    model.roi_heads.score_thresh = 0.6       # 你选的最佳阈值
    model.roi_heads.nms_thresh = 0.5         # 需要时 0.45~0.60 微调
    model.roi_heads.detections_per_img = 300
    model.to(device).eval()
    class_names = state.get("class_names", None)  # 可能为 None
    imgsz = state.get("imgsz", None)
    return model, class_names, imgsz

def filter_dets(out: dict, conf_thr: float, max_det: int):
    """按置信度阈值过滤，并截断前 max_det 个。"""
    scores = out["scores"]
    keep = scores >= conf_thr
    for k in ["boxes", "labels", "scores"]:
        out[k] = out[k][keep]
    if out["boxes"].numel() > 0 and len(out["boxes"]) > max_det:
        order = torch.argsort(out["scores"], descending=True)[:max_det]
        for k in ["boxes", "labels", "scores"]:
            out[k] = out[k][order]
    return out

def visualize(img_tensor: torch.Tensor,
              boxes_xyxy: torch.Tensor,
              labels: torch.Tensor,
              scores: torch.Tensor,
              class_names: Optional[List[str]],
              save_path: str):
    """把预测画到图片上保存（简单版，依赖 cv2 但可选）。"""
    try:
        import cv2
        import numpy as np
    except Exception:
        return  # 没装也不影响 JSON 输出

    img = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for b, c, s in zip(boxes_xyxy.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy()):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 175, 255), 2)
        cls_idx = int(c) - 1  # 我们训练时前景从1开始，这里显示减1更直观
        name = class_names[cls_idx] if class_names and 0 <= cls_idx < len(class_names) else f"id{c}"
        cv2.putText(img, f"{name}:{s:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 175, 255), 1, cv2.LINE_AA)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)

def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() and a.device == "cuda" else "cpu")

    # 数据集（保持与训练一致：labels=1..K，image_id=索引）
    ds = YoloDetectDataset(a.data_root, a.split, a.imgsz, get_transforms(False, a.imgsz))
    num_classes = ds.num_classes + 1

    # 加载模型
    model, class_names_from_ckpt, imgsz_ckpt = load_model(a.ckpt, num_classes, device)
    if class_names_from_ckpt is not None:
        class_names = class_names_from_ckpt
    else:
        class_names = getattr(ds, "names", None)

    # DataLoader
    ld = DataLoader(ds,
                    batch_size=a.batch,
                    shuffle=False,
                    num_workers=a.workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=(a.workers > 0))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.vis_dir:
        Path(a.vis_dir).mkdir(parents=True, exist_ok=True)

    # 推理
    t_sum, n_imgs, n_dets = 0.0, 0, 0
    preds = []

    scaler = torch.cuda.amp.autocast if (device.type == "cuda" and a.half) else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch_idx, (imgs, tgts) in enumerate(tqdm(ld, desc=f"infer {a.split}")):
            imgs = [im.to(device, non_blocking=True) for im in imgs]

            t0 = time.time()
            with scaler(dtype=torch.float16 if (device.type == "cuda" and a.half) else torch.bfloat16):
                outs = model(imgs)
            t_sum += time.time() - t0
            n_imgs += len(imgs)

            for i, (out, tgt) in enumerate(zip(outs, tgts)):
                out = {k: v.detach().cpu() for k, v in out.items()}
                out = filter_dets(out, a.conf_thr, a.max_det)

                # 统计
                n_dets += int(out["boxes"].shape[0])

                # 写 JSON（保持与训练一致：category_id为1..K；背景0跳过）
                xywh = box_convert(out["boxes"], in_fmt="xyxy", out_fmt="xywh").tolist()
                labels = out["labels"].tolist()
                scores = out["scores"].tolist()
                img_id = int(tgt["image_id"].item())

                for (x, y, w, h), c, s in zip(xywh, labels, scores):
                    if c == 0:
                        continue
                    preds.append({
                        "image_id": img_id,
                        "category_id": int(c),  # 1..K
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(s),
                    })

                # 可视化
                if a.vis_dir:
                    save_name = f"{a.split}_{img_id:06d}.jpg"
                    visualize(imgs[i].float().cpu(),
                              out["boxes"], out["labels"], out["scores"],
                              class_names, str(Path(a.vis_dir) / save_name))

    # 保存 JSON + 简单统计
    with open(a.out, "w") as f:
        json.dump(preds, f)
    avg_ms = (1000.0 * t_sum / max(n_imgs, 1))
    stats = {
        "images": n_imgs,
        "detections": n_dets,
        "avg_infer_ms": round(avg_ms, 2),
        "imgsz_arg": a.imgsz,
        "imgsz_ckpt": imgsz_ckpt,
        "half": bool(a.half and device.type == "cuda"),
        "conf_thr": a.conf_thr,
        "max_det": a.max_det,
    }
    with open(str(Path(a.out).with_suffix(".meta.json")), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {a.out}  |  avg_infer_ms={avg_ms:.2f}  |  imgs={n_imgs} dets={n_dets}")
    if a.vis_dir:
        print(f"Visualizations saved to: {a.vis_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/f_cnn/runs/best.pth")
    ap.add_argument("--split", choices=["valid", "test", "train"], default="valid")
    ap.add_argument("--out", default="/root/autodl-tmp/f_cnn/outputs/pred_frcnn_val.json")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # 新增：推理控制
    ap.add_argument("--conf_thr", type=float, default=0.05, help="score threshold for keeping detections")
    ap.add_argument("--max_det", type=int, default=300, help="max detections per image")
    ap.add_argument("--half", action="store_true", help="fp16 on CUDA for faster inference")
    ap.add_argument("--vis_dir", default="", help="optional: save visualized images to this dir")
    args = ap.parse_args()
    main(args)
