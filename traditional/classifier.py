from pathlib import Path
from typing import List, Tuple, Dict
import json
import joblib
import numpy as np
import cv2
from tqdm import tqdm
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .features import extract_feature
import time

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def walk_cls_dir(root_dir: Path) -> List[Tuple[Path, str]]:
    """return (img_path, class_name) list"""
    items = []
    for cls_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for p in cls_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, cls))
    return items

def build_label_map(items) -> Tuple[Dict[str,int], Dict[int,str]]:
    cls_names = sorted({c for _, c in items})
    name2id = {n:i for i,n in enumerate(cls_names)}
    id2name = {i:n for n,i in name2id.items()}
    return name2id, id2name

def load_features(root_dir: str, feature="hog",limit_per_class:int=200):
    root = Path(root_dir)
    items = walk_cls_dir(root)
    assert items, f"dir is empty:{root}"
    name2id, id2name = build_label_map(items)
    picked = {c: 0 for c in name2id.keys()}   #每类计数
    X, y = [], []
    for img_path, cls in tqdm(items, desc=f"Loading {root_dir}"):
        #每类限量
        if limit_per_class > 0 and picked[cls] >= limit_per_class:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        X.append(extract_feature(img, kind=feature))
        y.append(name2id[cls])
        picked[cls]+=1
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    print(f"!! per class counts:")
    print({k: v for k, v in sorted(picked.items())})
    return X, y, name2id, id2name

def train_svm(train_dir: str, model_path: str, labelmap_path: str, feature="hog"):
    #传入每类上限（先用200，可以调大/关掉）
    X, y, name2id, id2name = load_features(train_dir, feature, limit_per_class=-1)

    print(f"!!start SVC(probability=True), samples={len(y)}, dim={X.shape[1]}")
    t0 = time.time()
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        svm.SVC(kernel="linear", probability=True, cache_size=1000)  #大点cache更稳
    )
    clf.fit(X, y)   #这里会比较久 含Platt标定
    print(f"!!fit done in {time.time()-t0:.1f}s")

    joblib.dump(clf, model_path)
    with open(labelmap_path, "w", encoding="utf-8") as f:
        json.dump({"name2id": name2id, "id2name": id2name}, f, ensure_ascii=False, indent=2)
    print(f"!!model -> {model_path}, labelmap -> {labelmap_path}")
    return clf, id2name

def predict_dir(model_path: str, labelmap_path: str, img_dir: str, feature="hog"):
    clf = joblib.load(model_path)
    with open(labelmap_path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    id2name = {int(k):v for k,v in maps["id2name"].items()}

    preds = []
    for p in Path(img_dir).rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        feat = extract_feature(img, kind=feature).reshape(1, -1)
        prob = clf.predict_proba(feat)[0]
        label_id = int(np.argmax(prob))
        score = float(prob[label_id])
        H, W = img.shape[:2]
        preds.append({
            "image_id": p.name,
            "box": [0,0,W,H],         #先用整图占位 接入候选框后替换
            "score": score,
            "label": label_id,
            "label_name": id2name[label_id],
        })
    return preds
