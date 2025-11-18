# 把 data/AgroPest-12 下的 YOLO 数据，整理为 data/agropest12_classic/<ClassName>/*.jpg
from pathlib import Path
import shutil, yaml

YAML_PATH = Path("./data/AgroPest-12/data.yaml")  #现在的路径
assert YAML_PATH.exists(), f"data.yaml no exist：{YAML_PATH}"

cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
names = cfg["names"]
root = YAML_PATH.parent
train_img = (root / cfg["test"]).resolve()               # .../AgroPest-12/train/images
train_lbl = (root / cfg["test"].replace("images","labels")).resolve()

out_root = Path("data/agropest12_classic_test").resolve()
out_root.mkdir(parents=True, exist_ok=True)
for n in names:
    (out_root / n).mkdir(exist_ok=True)

count = 0
for lbl in train_lbl.glob("*.txt"):
    stem = lbl.stem
    #兼容jpg/jpeg/png
    for ext in (".jpg",".jpeg",".png"):
        img = train_img / f"{stem}{ext}"
        if img.exists():
            break
    else:
        continue

    lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
    if not lines: 
        continue
    cls_id = int(lines[0].split()[0])     # 多目标时取第一行类别即可跑通
    dst = out_root / names[cls_id] / img.name
    if not dst.exists():
        shutil.copy(img, dst)
        count += 1

print(f"have copy {count} to {out_root}")
