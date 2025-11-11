# f_cnn/eval_frcnn.py
# 评估 Faster R-CNN：VOC07 mAP@0.5（11-point），并输出每类 AP/GT/Det、整体 P/R
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from f_cnn.data import YoloDetectDataset, get_transforms, collate_fn
from f_cnn.train_frcnn import build_model


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,4], b: [M,4]
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return (inter / union).clamp(0, 1)


def voc07_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return float(ap)


def evaluate(model, loader, device, class_names, iou_thr=0.5, score_thr=0.001, max_det=300):
    K = len(class_names)
    det_scores = [[] for _ in range(K)]   # 每类：所有检测的得分
    det_tp     = [[] for _ in range(K)]   # 每类：对应 TP/FP 标记（与 det_scores 对齐）
    gt_count   = [0  for _ in range(K)]   # 每类 GT 数

    model.eval()
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            outs = model(imgs)
            for out, tgt in zip(outs, tgts):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                # 统计 GT（忽略背景 0）
                valid_gt = gt_labels > 0
                gt_boxes = gt_boxes[valid_gt]
                gt_labels = gt_labels[valid_gt]
                for c in gt_labels.tolist():
                    if 1 <= c <= K:
                        gt_count[c - 1] += 1

                boxes  = out["boxes"].to(device)
                scores = out["scores"].to(device)
                labels = out["labels"].to(device)

                # 过滤低分与 top-k
                keep = scores >= score_thr
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if boxes.shape[0] == 0:
                    continue
                if boxes.shape[0] > max_det:
                    order = torch.argsort(scores, descending=True)[:max_det]
                    boxes, scores, labels = boxes[order], scores[order], labels[order]

                # 分类别匹配
                for c in torch.unique(labels).tolist():
                    if c == 0 or c > K:
                        continue
                    pi = (labels == c).nonzero(as_tuple=True)[0]
                    pb, ps = boxes[pi], scores[pi]
                    gi = (gt_labels == c).nonzero(as_tuple=True)[0]
                    gb = gt_boxes[gi]
                    used = torch.zeros((gb.shape[0],), dtype=torch.bool, device=device)

                    # 按得分排序
                    order = torch.argsort(ps, descending=True)
                    pb, ps = pb[order], ps[order]

                    ious = iou_xyxy(pb, gb) if gb.numel() else torch.zeros((pb.shape[0], 0), device=device)
                    for j in range(pb.shape[0]):
                        ok = 0
                        if gb.numel():
                            best_i = torch.argmax(ious[j]).item()
                            best   = ious[j, best_i].item()
                            if best >= iou_thr and not used[best_i]:
                                ok = 1
                                used[best_i] = True
                        det_scores[c - 1].append(ps[j].item())
                        det_tp[c - 1].append(ok)

    # 汇总指标
    per_class = []
    aps = []
    total_tp = total_fp = total_gt = 0
    for c in range(K):
        sc = np.array(det_scores[c], dtype=np.float32)
        tp = np.array(det_tp[c], dtype=np.int32)
        npos = int(gt_count[c])
        if sc.size == 0:
            ap, prec, rec = 0.0, np.array([0.0]), np.array([0.0])
            tp_c = 0; fp_c = 0
        else:
            order = np.argsort(-sc)
            tp = tp[order]
            fp = 1 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            rec  = tp_cum / max(npos, 1)
            ap   = voc07_ap(rec, prec)
            tp_c = int(tp_cum[-1]); fp_c = int(fp_cum[-1])
        aps.append(ap)
        total_tp += tp_c
        total_fp += fp_c
        total_gt += npos
        per_class.append({
            "class": class_names[c],
            "AP50": round(float(ap), 4),
            "GT": npos,
            "Det": int(sc.size),
        })

    mAP = float(np.mean(aps)) if aps else 0.0
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_gt, 1)
    summary = {"mAP50": round(mAP, 4), "precision": round(float(precision), 4),
               "recall": round(float(recall), 4), "gt_total": int(total_gt)}

    return summary, per_class


def main(a):
    device = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    root = Path(a.data_root)

    # 验证集
    val_set = YoloDetectDataset(str(root), a.split, a.imgsz, get_transforms(False, a.imgsz))
    class_names = getattr(val_set, "names", None)
    assert class_names and len(class_names) > 0, "class names not found"
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=a.workers,
                        pin_memory=True, collate_fn=collate_fn)

    # 模型
    num_classes = len(class_names) + 1
    model = build_model(num_classes, backbone=a.backbone)
    ckpt = torch.load(a.weights, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    summary, table = evaluate(
        model, loader, device, class_names,
        iou_thr=0.5, score_thr=a.conf, max_det=a.max_det
    )

    print("\n===== Eval @IoU=0.5 =====")
    for row in table:
        print(f"{row['class']:>15s}  AP50={row['AP50']:.4f}  GT={row['GT']:3d}  Det={row['Det']:3d}")
    print("\nSummary:", summary)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "per_class": table}, f, indent=2, ensure_ascii=False)
        print("Saved:", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backbone", choices=["resnet50", "mobile"], default="resnet50")
    ap.add_argument("--conf", type=float, default=0.001, help="score阈值（评估建议很低）")
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--split", choices=["valid", "test", "train"], default="valid")
    ap.add_argument("--out", default="")
    main(ap.parse_args())
