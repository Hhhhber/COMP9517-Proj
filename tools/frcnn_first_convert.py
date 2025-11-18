# tools/frcnn_first_convert.py
import os
import json
import yaml
from pathlib import Path

FRCNN_JSON = "./outputs/frcnn/preds_frcnn_test.json"
DATA_YAML  = "./data/AgroPest-12/data.yaml"
IMG_DIR = "./data/AgroPest-12/test/images"
OUT_JSON   = "./outputs/frcnn/frcnn_test_middle.json"

# 1. 读类别名（主要是为了防止 raw 里没有 label_name 的情况）
with open(DATA_YAML, "r") as f:
    y = yaml.safe_load(f)

names = y.get("names")
if isinstance(names, dict):
    names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]

# 2. 扫描图片目录，按排序建立 id -> 文件名 映射
files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

id2name = {i: files[i] for i in range(len(files))}
print(f"success, we have {len(id2name)} pictures, the five ahead：", list(id2name.items())[:5])

# 3. 读取 raw FRCNN json（image_id 是 '0' / '1' 这种字符串）
with open(FRCNN_JSON, "r") as f:
    frcnn_data = json.load(f)

out = []

for det in frcnn_data:
    img_id_raw = det["image_id"]           # 例如 "0"
    # 如果是数字字符串，就用映射；否则直接当成文件名
    if isinstance(img_id_raw, int) or (isinstance(img_id_raw, str) and img_id_raw.isdigit()):
        idx = int(img_id_raw)
        img_name = id2name.get(idx, str(img_id_raw))
    else:
        img_name = img_id_raw

    box = det["box"]                       # 已经是 [x1,y1,x2,y2]
    score = float(det["score"])
    label = int(det["label"])

    # label_name：优先用原 json 里的，没有就根据 names 补
    label_name = det.get("label_name")
    if label_name is None and names is not None and 0 <= label < len(names):
        label_name = names[label]

    out.append({
        "image_id": img_name,
        "box": box,
        "score": score,
        "label": label,
        "label_name": label_name,
    })

# 4. 写出新 json
out_path = Path(OUT_JSON)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"Done! Saved to {out_path}, total {len(out)} detections.")