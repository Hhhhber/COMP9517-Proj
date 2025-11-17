# -*- coding: utf-8 -*-
# 使用文件名前缀(ants-, bees-, ...)推断真值 评估preds_trad_valid.json
import argparse, json, re, pandas as pd, numpy as np
import matplotlib.pyplot as plt

PREFIX2CLASS = {
    "ants": "Ants",
    "bees": "Bees",
    "beetle": "Beetles",
    "earwig": "Earwigs",
    "caterpillar": "Caterpillars",
    "catterpillar": "Caterpillars",  #兼容两种拼法
    "earthworm": "Earthworms",
    "earthworms": "Earthworms",      #有些文件可能带复数
    "grasshopper": "Grasshoppers",
    "moth": "Moths",
    "snail": "Snails",
    "slug": "Slugs",
    "wasp": "Wasps",
    "weevil": "Weevils",
}


def infer_true_label(filename: str):
    m = re.match(r"([a-zA-Z]+)-", filename)
    if not m:
        return None
    pref = m.group(1).lower()
    return PREFIX2CLASS.get(pref)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="./outputs/traditional/preds.json")
    ap.add_argument("--out_prefix", default="./outputs/plot_eval/")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    rows, bad = [], 0
    for p in preds:
        fname = p["image_id"]
        true_name = infer_true_label(fname)
        if true_name is None:
            bad += 1
            continue
        rows.append({
            "image_id": fname,
            "true": true_name,
            "pred": p.get("label_name"),
            "score": p.get("score", None),
        })

    if not rows:
        print("No evaluation samples are available. Please check if the filenames have a prefix (such as ants- or bees-).")
        return

    import pandas as pd, numpy as np
    df = pd.DataFrame(rows)
    labels = sorted({*df["true"].unique(), *df["pred"].dropna().unique()})
    idx = {c:i for i,c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, pr in zip(df["true"], df["pred"]):
        if pr in idx:
            cm[idx[t], idx[pr]] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else float("nan")

    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    support   = cm.sum(axis=1).astype(int)

    report = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }, index=labels).sort_index()

    #导出CSV
    rep_csv = f"{args.out_prefix}eval_report.csv"
    cm_csv  = f"{args.out_prefix}confusion_matrix.csv"
    pred_csv= f"{args.out_prefix}predictions_with_truth.csv"
    report.to_csv(rep_csv, index=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv, index=True)
    df.to_csv(pred_csv, index=False)

    #画图（混淆矩阵）
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Confusion Matrix (valid)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    fig_png = f"{args.out_prefix}confusion_matrix.png"
    plt.savefig(fig_png, dpi=180)
    plt.close()

    print(f"samples sizes: {total} | correct number: {int(np.trace(cm))} | accuracy={acc:.4f}")
    print(f"outputs: {rep_csv}, {cm_csv}, {pred_csv}, {fig_png}")
    if bad:
        print(f" {bad} samples could not be parsed from their filenames (missing prefix or not in the mapping).")

if __name__ == "__main__":
    main()
