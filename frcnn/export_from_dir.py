import argparse, json
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

@torch.no_grad()
def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location="cpu")
    # 从权重里读 num_classes / class_names，避免不匹配
    class_names = state.get("class_names", None)
    if class_names and isinstance(class_names, dict):
        # 可能是 id->name 或 name->id，统一为 id->name
        if all(isinstance(k, int) for k in class_names.keys()):
            id2name = class_names
        else:
            id2name = {v: k for k, v in class_names.items()}
    elif class_names and isinstance(class_names, (list, tuple)):
        id2name = {i+1: n for i, n in enumerate(class_names)}  # torchvision: label 从1开始
    else:
        id2name = None

    num_classes = state.get("num_classes", None)
    if num_classes is None:
        num_classes = (max(id2name.keys())+1) if id2name else None
    if num_classes is None:
        raise ValueError("无法从权重中推断 num_classes，请在命令行传 --num_classes")

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    return model, id2name

def list_images(p: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for f in sorted(p.rglob("*")):
        if f.suffix.lower() in exts:
            yield f

def main(a):
    device = torch.device(a.device if torch.cuda.is_available() and a.device=="cuda" else "cpu")
    model, id2name = load_model(a.weights, device)
    if id2name is None:
        # 兜底：1..N -> "class_1"...（注意0是背景）
        id2name = {i: f"class_{i}" for i in range(1, model.roi_heads.box_predictor.cls_score.out_features)}

    tfm = T.Compose([T.ToTensor()])  # Faster R-CNN 期望 0~1 tensor

    results = []
    imgs_dir = Path(a.images_dir)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in list_images(imgs_dir):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        t = tfm(img).to(device)
        out = model([t])[0]  # dict: boxes, labels, scores

        score, label = 0.0, -1
        if len(out["scores"]) > 0:
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()
            keep = scores >= a.conf
            if keep.any():
                scores = scores[keep]; labels = labels[keep]
            if len(scores) > 0:
                top = int(torch.argmax(scores))
                score = float(scores[top])
                label = int(labels[top])  # 1..N；0 是背景

        label_name = id2name.get(label, "__background__") if label > 0 else "__background__"
        box_full = [0, 0, W, H]  # 整图占位

        results.append({
            "image_id": p.name,
            "box": box_full,
            "score": round(score, 6),
            "label": label if label > 0 else -1,   # 与你们约定：无检测用 -1
            "label_name": label_name
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} preds to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="要遍历的图片文件夹（如 data/test/images）")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_classes", type=int, default=None, help="若权重未记录类数可手动指定")
    a = ap.parse_args()
    main(a)
