# test.py
from ultralytics import YOLO
import numpy as np
import os, glob, sys

# ===== 可按需修改的配置 =====
DATA     = "/root/autodl-tmp/dataset/data.yaml"                         # data.yaml 路径
BEST     = "/root/autodl-tmp/yolo/runs/detect/train/weights/best.pt"    # 你当前训练的best.pt
IMG_SIZE = 256
BATCH    = 32
DEVICE   = 0        # 没GPU就改成 None 或删掉 device 参数

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
    """优先在 test 集评估；没有 test 就回退到 val"""
    kwargs = dict(data=DATA, imgsz=IMG_SIZE, batch=BATCH, conf=0.001)
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    try:
        print("[Info] 在 test 集评估...")
        return model.val(split="test", **kwargs)
    except Exception as e:
        print(f"[Warn] 使用 test 集失败（可能 data.yaml 未配置 test）：{e}")
        print("[Info] 回退到 val 集评估...")
        return model.val(split="val", **kwargs)

def main():
    model = load_model()
    val   = evaluate(model)

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

if __name__ == "__main__":
    main()
