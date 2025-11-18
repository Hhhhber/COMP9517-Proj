# test.py
from ultralytics import YOLO
import numpy as np
import os, glob, sys
import json
import yaml
from pathlib import Path
from PIL import Image

# ===== Configuration =====
DATA     = "./data/AgroPest-12/data.yaml"     # path to data.yaml
BEST     = "./outputs/deep_yolo/train/weights/best.pt"    # best.pt
IMG_SIZE = 256
BATCH    = 32
DEVICE   = None        # if no GPU we set None

# output path
OUT_JSON_TPL = "./outputs/deep_yolo/preds_yolo_{split}_plain.json"


def latest_best():
    """find best.pt from outputs dir"""
    cands = glob.glob("./outputs/deep_yolo/train/weights/best.pt")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load_model():
    best = BEST
    if not os.path.exists(best):
        alt = latest_best()
        if alt is None:
            print(f"[Error] weight not found：{BEST}，and no path like './outputs/deep_yolo/train/weights/best.pt'")
            sys.exit(1)
        print(f"[Info] no path，use the last weight：{alt}")
        best = alt
    else:
        print(f"[Info] use the weight：{best}")
    return YOLO(best)


def evaluate(model):
    """evaluate on test set"""
    kwargs = dict(data=DATA, imgsz=IMG_SIZE, batch=BATCH, conf=0.001)
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    try:
        print("[Info] evaluate on test set...")
        val = model.val(split="test", **kwargs)
        return val, "test"
    except Exception as e:
        print(f"[Warn] fail on using test set：{e}")
        print("[Info] turn into valid evaluation...")
        val = model.val(split="val", **kwargs)
        return val, "val"


def save_preds_json(model, split, out_path):
    with open(DATA, "r") as f:
        y = yaml.safe_load(f)

    img_source = y.get(split)
    if img_source is None:
        raise RuntimeError(f"data.yaml has no '{split}'，please check {DATA}")

    if not os.path.isabs(img_source):
        img_source = str((Path(DATA).parent / img_source).resolve())

    print(f"[Info] start with split='{split}' generate predict json，pic dir：{img_source}")

    # run the prediction
    pred_kwargs = dict(
        source=img_source,
        imgsz=IMG_SIZE,
        conf=0.001,
        verbose=False,
    )
    if DEVICE is not None:
        pred_kwargs["device"] = DEVICE

    results = model.predict(**pred_kwargs)

    # collect the results
    all_out = []
    for r in results:
        img_path = r.path
        im = Image.open(img_path)
        w, h = im.size

        boxes = r.boxes
        preds = []
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().tolist()
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

    # write the document
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)

    print(f"[Info] have saved {len(all_out)} predictions of pic to：{out_path}")


def main():
    model = load_model()
    val, split_used = evaluate(model)

    # ------ 指标提取 ------
    m   = val.results_dict
    P   = float(m.get("metrics/precision(B)", 0.0))
    R   = float(m.get("metrics/recall(B)",    0.0))
    F1  = 2 * P * R / (P + R + 1e-12)
    m50 = float(m.get("metrics/mAP50(B)",     0.0))
    m95 = float(m.get("metrics/mAP50-95(B)",  0.0))

    # 混淆矩阵求 Accuracy
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
            out_path = OUT_JSON_TPL.format(split=split_used)  # preds_yolo_test_plain.json
            save_preds_json(model, split_used, out_path)
    except Exception as e:
        print(f"[Warn] fail in generating json：{e}")


if __name__ == "__main__":
    main()
