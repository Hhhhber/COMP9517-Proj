import json
import argparse
from pathlib import Path

"""
支持两种输入格式：

1）你现在的 plain 格式（按图片）：
[
  {
    "image": "xxx.jpg",
    "width": 640,
    "height": 640,
    "preds": [
      {"cls": 0, "score": 0.9, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
      ...
    ]
  },
  ...
]

2）旧版脚本注释里的格式（按图片）：
[
  {
    "image_id": "xxx.jpg",
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.9, 0.8, ...],
    "labels": [0, 3, ...]
  },
  ...
]

输出统一格式（按框）：
[
  {
    "image_id": "xxx.jpg",
    "box": [x1, y1, x2, y2],
    "score": 0.9,
    "label": 0
  },
  ...
]
"""


def main(a):
    in_path = Path(a.yolo_json)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    unified = []

    if isinstance(preds, list) and preds:
        first = preds[0]

        # ====== 情况 1：你的 plain 格式（image + preds） ======
        if "image" in first and "preds" in first:
            for img_rec in preds:
                img_id = img_rec["image"]
                W = img_rec.get("width", 1)
                H = img_rec.get("height", 1)
        
                preds_list = img_rec.get("preds", [])
                if not preds_list:
                    continue
        
                # 只取 score 最高的那个框
                p = max(preds_list, key=lambda d: float(d.get("score", 0.0)))
        
                # 归一化中心点 xywh -> 像素坐标的 x1,y1,x2,y2
                xc = float(p["x"])
                yc = float(p["y"])
                ww = float(p["w"])
                hh = float(p["h"])
        
                xc_abs = xc * W
                yc_abs = yc * H
                w_abs = ww * W
                h_abs = hh * H
        
                x1 = xc_abs - w_abs / 2.0
                y1 = yc_abs - h_abs / 2.0
                x2 = xc_abs + w_abs / 2.0
                y2 = yc_abs + h_abs / 2.0
        
                unified.append(
                    {
                        "image_id": img_id,
                        "box": [x1, y1, x2, y2],
                        "score": float(p["score"]),
                        "label": int(p["cls"]),
                    }
                )

        # ====== 情况 2：旧格式（image_id + boxes + scores + labels） ======
        elif "image_id" in first and "boxes" in first:
            for img_rec in preds:
                img_id = img_rec["image_id"]
                boxes = img_rec["boxes"]
                scores = img_rec["scores"]
                labels = img_rec["labels"]
                for b, s, c in zip(boxes, scores, labels):
                    unified.append(
                        {
                            "image_id": img_id,
                            "box": b,  # 已经是 [x1, y1, x2, y2]
                            "score": float(s),
                            "label": int(c),
                        }
                    )
        else:
            print(f"[Warn] JSON 格式不在两种支持范围内，首个元素 keys: {list(first.keys())}")
    else:
        print(f"[Warn] 输入 json 为空或格式异常：{in_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)

    print(f"Saved unified detections to: {out_path}  (total {len(unified)} records)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_json", required=True, help="YOLO 检测结果 json（plain 或旧格式都行）")
    ap.add_argument("--out", required=True, help="输出统一格式 json 路径")
    args = ap.parse_args()
    main(args)
