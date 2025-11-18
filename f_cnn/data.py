import glob
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A


def load_names_from_yaml(data_yaml: str):
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names, y


def build_augs(train: bool, imgsz: int, big_object: bool = True):
    """big_object=True：目标较大数据集 -> 缩放/旋转更温和。"""
    padding_color = (114, 114, 114)
    if train:
        scale_limit = 0.10 if big_object else 0.20
        rotate_limit = 8 if big_object else 12
        return A.Compose(
            [
                A.LongestMaxSize(max_size=imgsz),
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0, value=padding_color),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=scale_limit,
                    rotate_limit=rotate_limit,
                    border_mode=0,
                    value=padding_color,
                    p=0.5,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=imgsz),
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0, value=padding_color),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


class YoloDetectDataset(Dataset):
    """
    目录：
      ./data/AgroPest-12/{train,valid,test}/images/*.jpg|png|jpeg|bmp|tif|webp
      ./data/AgroPest-12/{train,valid,test}/labels/*.txt
      ./data/AgroPest-12/data.yaml
    YOLO标签：class cx cy w h（归一化）
    """
    def __init__(self, root: str, split: str = "train", img_size: int = 1024,
                 transforms=None, big_object: bool = True):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"
        assert self.img_dir.exists(), f"Missing {self.img_dir}"
        assert self.lbl_dir.exists(), f"Missing {self.lbl_dir}"

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
        paths = []
        for e in exts: paths.extend(glob.glob(str(self.img_dir / e)))
        self.img_paths = sorted(paths)

        self.names, self.meta = load_names_from_yaml(str(self.root / "data.yaml"))
        self.num_classes = len(self.names)
        self.img_size = img_size

        self.augs = transforms if transforms is not None else build_augs(split == "train", img_size, big_object)

    def __len__(self):
        return len(self.img_paths)

    def _read_yolo_txt(self, lp: Path, w: int, h: int):
        boxes, labels = [], []
        if not lp.exists(): return boxes, labels
        with open(lp, "r") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                parts = s.split()
                if len(parts) < 5: continue
                c, xc, yc, bw, bh = map(float, parts[:5])
                x_c, y_c = xc * w, yc * h
                bw, bh = bw * w, bh * h
                x1, y1 = x_c - bw / 2, y_c - bh / 2
                x2, y2 = x_c + bw / 2, y_c + bh / 2
                if (x2 - x1) <= 1 or (y2 - y1) <= 1:  # 过滤极小/异常框
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c) + 1)  # Faster R-CNN: 前景 1..K（0 留给背景）
        return boxes, labels

    def __getitem__(self, idx: int):
        ip = Path(self.img_paths[idx])
        img = Image.open(ip).convert("RGB")      # 强制 RGB
        w, h = img.size

        lp = self.lbl_dir / (ip.stem + ".txt")
        boxes, labels = self._read_yolo_txt(lp, w, h)

        # Albumentations 期望 numpy；处理灰度/RGBA；确保 uint8 & 3通道
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        img_np = img_np.astype(np.uint8)

        bxs = np.array(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.zeros((0, 4), np.float32)
        lbs = np.array(labels, dtype=np.int64) if len(labels) else np.zeros((0,), np.int64)

        transformed = self.augs(image=img_np, bboxes=bxs.tolist(), labels=lbs.tolist())
        im_np = transformed["image"]            # HWC, uint8
        tb = transformed.get("bboxes", [])
        tl = transformed.get("labels", [])

        # 手动转成 Torch Tensor（CHW，float32，0-1）
        if im_np.dtype != np.uint8:
            im_np = im_np.astype(np.uint8)
        img_t = torch.from_numpy(im_np).permute(2, 0, 1).contiguous().float() / 255.0

        if len(tb) == 0:
            bx_t = torch.zeros((0, 4), dtype=torch.float32)
            lb_t = torch.zeros((0,), dtype=torch.int64)
        else:
            bx_t = torch.tensor(tb, dtype=torch.float32).view(-1, 4)
            lb_t = torch.tensor(tl, dtype=torch.int64)

        target = {
            "boxes": bx_t,
            "labels": lb_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transforms(train: bool = True, img_size: int = 1024):
    # 兼容旧接口
    return build_augs(train, img_size, big_object=True)
