# test.py
from ultralytics import YOLO
import numpy as np
import os, glob, sys
import json
import yaml
from pathlib import Path
from PIL import Image

# ===== 可按需修改的配置 =====
DATA     = "/root/autodl-tmp/dataset/data.yaml"                         # data.yaml 路径
BEST     = "/root/autodl-tmp/yolo/runs/detect/train/weights/best.pt"    # 你当前训练的best.pt
IMG_SIZE = 256
BATCH    = 32
DEVICE   = 0        # 没GPU就改成 None 或删掉 device 参数

# 这里设定你要输出的 json 路径（可以按需修改）
OUT_JSON_TPL = "/root/autodl-tmp/outputs88/preds_yolo_{split}_plain.json"


def latest_best():
    """从 runs/detect/*/weights/ 中找最近一次训练的 best.pt"""
    cands = glob.glob("runs/detect/*/weights/best.pt")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load_model():
    best = BEST
    if not os.path.exists(best):
        alt = latest_best()
        if alt is None:
            print(f"[Error] 找不到权重：{BEST}，且 runs/detect/*/weights/best.pt 也不存在")
            sys.exit(1)
        print(f"[Info] 指定路径不存在，自动改用最近的权重：{alt}")
        best = alt
    else:
        print(f"[Info] 使用权重：{best}")
    return YOLO(best)


def evaluate(model):
    """优先在 test 集评估；没有 test 就回退到 val，并把实际使用的 split 一起返回"""
    kwargs = dict(data=DATA, imgsz=IMG_SIZE, batch=BATCH, conf=0.001)
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    try:
        print("[Info] 在 test 集评估...")
        val = model.val(split="test", **kwargs)
        return val, "test"
    except Exception as e:
        print(f"[Warn] 使用 test 集失败（可能 data.yaml 未配置 test）：{e}")
        print("[Info] 回退到 val 集评估...")
        val = model.val(split="val", **kwargs)
        return val, "val"


def save_preds_json(model, split, out_path):
    """
    对指定 split（'test' 或 'val'）整个数据集跑一遍预测，并保存为 json。

    输出格式：
    [
      {
        "image": "xxxx.jpg",
        "width": W,
        "height": H,
        "preds": [
          {"cls": int, "score": float, "x": xc, "y": yc, "w": bw, "h": bh}, ...
        ]
      },
      ...
    ]
    坐标是归一化 xywh（和 YOLO txt 一致），方便后面给 yolo_to_unified.py 用。
    """
    # 1) 从 data.yaml 里拿到这个 split 对应的图片路径
    with open(DATA, "r") as f:
        y = yaml.safe_load(f)

    img_source = y.get(split)
    if img_source is None:
        raise RuntimeError(f"data.yaml 中没有 '{split}' 字段，请检查 {DATA}")

    # 相对路径 -> 变成绝对路径
    if not os.path.isabs(img_source):
        img_source = str((Path(DATA).parent / img_source).resolve())

    print(f"[Info] 开始为 split='{split}' 生成预测 json，图片目录：{img_source}")

    # 2) 跑预测
    pred_kwargs = dict(
        source=img_source,
        imgsz=IMG_SIZE,
        conf=0.001,   # 阈值放低一点，把所有框都导出来
        verbose=False,
    )
    if DEVICE is not None:
        pred_kwargs["device"] = DEVICE

    results = model.predict(**pred_kwargs)

    # 3) 收集结果
    all_out = []
    for r in results:
        img_path = r.path
        im = Image.open(img_path)
        w, h = im.size

        boxes = r.boxes
        preds = []
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().tolist()   # 归一化 xywh
            clses = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for (xc, yc, bw, bh), c, s in zip(xywhn, clses, confs):
                preds.append(
                    {
                        "cls": int(c),
                        "score": float(s),
                        "x": float(xc),
                        "y": float(yc),
                        "w": float(bw),
                        "h": float(bh),
                    }
                )

        all_out.append(
            {
                "image": os.path.basename(img_path),
                "width": w,
                "height": h,
                "preds": preds,
            }
        )

    # 4) 写文件
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)

    print(f"[Info] 已保存 {len(all_out)} 张图片的预测到：{out_path}")


def main():
    model = load_model()
    val, split_used = evaluate(model)

    # ------ 指标提取（与你原版保持一致）------
    m   = val.results_dict
    P   = float(m.get("metrics/precision(B)", 0.0))
    R   = float(m.get("metrics/recall(B)",    0.0))
    F1  = 2 * P * R / (P + R + 1e-12)
    m50 = float(m.get("metrics/mAP50(B)",     0.0))
    m95 = float(m.get("metrics/mAP50-95(B)",  0.0))

    # 混淆矩阵求 Accuracy（作为参考）
    cm = getattr(val, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix"):
        cm_mat = cm.matrix
        ACC = float(np.trace(cm_mat) / (cm_mat.sum() + 1e-12))
    else:
        ACC = float("nan")

    # ------ 打印结果 ------
    print("\n====== Eval Result ======")
    print(f"Test Precision:      {P:.4f}")
    print(f"Test Recall:         {R:.4f}")
    print(f"Test F1:             {F1:.4f}")
    print(f"Test Accuracy(cm):   {ACC:.4f}")
    print(f"Test mAP@0.5:        {m50:.4f}")
    print(f"Test mAP@0.5:0.95:   {m95:.4f}")
    print("=========================\n")

    # ------ 额外：生成 json ------
    try:
            out_path = OUT_JSON_TPL.format(split=split_used)  # 比如 preds_yolo_test_plain.json
            save_preds_json(model, split_used, out_path)
    except Exception as e:
        print(f"[Warn] 生成预测 json 失败：{e}")


if __name__ == "__main__":
    main()
