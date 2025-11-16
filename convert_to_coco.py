#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, csv, argparse, random
from PIL import Image

def img_size(p):
    with Image.open(p) as im: return im.size

def clip(v, lo, hi): return max(lo, min(hi, v))
def xyxy2xywh(x1,y1,x2,y2): return [float(x1),float(y1),float(x2-x1),float(y2-y1)]

def dump(coco, path):
    with open(path,"w",encoding="utf-8") as f: json.dump(coco,f,ensure_ascii=False,indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)          # 图片根目录
    ap.add_argument("--csv", required=True)             # 标注CSV：file_name,xmin,ymin,xmax,ymax,label
    ap.add_argument("--out_dir", default="data")        # 输出目录
    ap.add_argument("--splits", default="7,1,2")        # 训练/验证/测试比例
    ap.add_argument("--min_box", type=float, default=2) # 最小框像素
    ap.add_argument("--seed", type=int, default=9517)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # 读取CSV
    rows = []
    with open(a.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"file_name","xmin","ymin","xmax","ymax","label"}
        if not need.issubset(r.fieldnames):
            raise SystemExit(f"CSV列需包含: {need}，当前列: {r.fieldnames}")
        rows = [x for x in r]

    # 按图片聚合
    per_img, labels, paths = {}, set(), {}
    for r in rows:
        fn = r["file_name"]; p = os.path.join(a.images, fn)
        if not os.path.exists(p):  # 容错：按文件名在根下直接找
            p = os.path.join(a.images, os.path.basename(fn))
        if not os.path.exists(p): 
            continue
        w,h = img_size(p)
        try:
            x1,y1,x2,y2 = map(float,[r["xmin"],r["ymin"],r["xmax"],r["ymax"]])
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        x1,y1 = clip(x1,0,w-1), clip(y1,0,h-1)
        x2,y2 = clip(x2,1,w),   clip(y2,1,h)
        box = xyxy2xywh(x1,y1,x2,y2)
        if box[2] < a.min_box or box[3] < a.min_box: 
            continue
        per_img.setdefault(p,[]).append((box, r["label"]))
        labels.add(str(r["label"]).strip()); paths[p]= (w,h)

    if not per_img: raise SystemExit("没有有效标注，请检查CSV/图片路径。")
    names = sorted(labels); id_of = {n:i+1 for i,n in enumerate(names)}

    # 切分
    rng = random.Random(a.seed)
    imgs = list(per_img.keys()); rng.shuffle(imgs)
    s = list(map(int, a.splits.split(","))); t=sum(s)
    if t <= 0: raise SystemExit("--splits 必须为正整数, 例如 7,1,2")
    n=len(imgs); ntr=n*s[0]//t; nv=n*s[1]//t
    split = {"train":imgs[:ntr], "val":imgs[ntr:ntr+nv], "test":imgs[ntr+nv:]}

    # 生成COCO
    ann_id = 1; img_id = 1
    for k,v in split.items():
        coco = {"images":[], "annotations":[], "categories":[{"id":id_of[n],"name":n} for n in names]}
        for p in v:
            w,h = paths[p]
            coco["images"].append({"id":img_id,"file_name":os.path.relpath(p,a.images),"width":w,"height":h})
            for box,label in per_img[p]:
                coco["annotations"].append({
                    "id":ann_id,"image_id":img_id,"category_id":id_of[str(label).strip()],
                    "bbox":box,"area":float(box[2]*box[3]),"iscrowd":0
                })
                ann_id += 1
            img_id += 1
        dump(coco, os.path.join(a.out_dir, f"{k}.json"))
    # 类别映射便于下游统一
    with open(os.path.join(a.out_dir,"class_mapping.json"),"w",encoding="utf-8") as f:
        json.dump(id_of,f,ensure_ascii=False,indent=2)
    print("[OK] 输出:")
    for name in ("train.json","val.json","test.json","class_mapping.json"):
        print(" -", os.path.join(a.out_dir, name))

if __name__ == "__main__":
    main()