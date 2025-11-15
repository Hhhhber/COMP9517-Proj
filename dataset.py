#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py — COCODetectionDataset (PyTorch)
------------------------------------------

A lightweight PyTorch Dataset for COCO-style detection JSON, returning
(images, targets) pairs suitable for Faster R-CNN, YOLO-style trainers, etc.

JSON requirements (COCO-style):
- "images": [{"id": int, "file_name": "path/relative/to/image_root.jpg", "width": int, "height": int}, ...]
- "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x_min, y_min, width, height], "iscrowd": 0}, ...]
- "categories": [{"id": int, "name": "class_name"}, ...]

Target dict returned per sample:
- "image_id": 1D int64 tensor, length 1
- "boxes": (N, 4) float32 tensor with xyxy boxes
- "labels": (N,) int64 tensor (category_id)
- Optionally "area" and "iscrowd" if present in JSON

Usage:
------
from torch.utils.data import DataLoader
from dataset import COCODetectionDataset, detection_collate

ds = COCODetectionDataset(json_path="data/train.json", image_root="data/AgroPest-12/images")
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=detection_collate)

for images, targets in dl:
    # images: List[PIL.Image.Image] (or tensors if you transform them)
    # targets: List[Dict[str, Tensor]]
    ...

Notes:
------
- Transforms: pass a callable transforms(img, target) -> (img, target).
  You can plug in torchvision transforms or Albumentations wrappers.
- If your JSON "file_name" are absolute paths, set image_root to "".
- This dataset does not depend on torchvision; it only uses PIL and torch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import json
from PIL import Image
import torch
from torch.utils.data import Dataset


def _to_xyxy(box_xywh):
    """Convert [x_min, y_min, w, h] -> [x1, y1, x2, y2]."""
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        image_root: str | Path,
        transforms: Optional[Callable] = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            json_path: Path to COCO-style JSON (train/val/test).
            image_root: Root directory of images referenced by JSON "file_name".
            transforms: Optional callable (img, target) -> (img, target).
            strict: If True, raises on missing files; otherwise silently skips during __getitem__ (rare).
        """
        self.json_path = Path(json_path)
        self.image_root = Path(image_root) if image_root else Path(".")
        self.transforms = transforms
        self.strict = strict

        with open(self.json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Basic keys
        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])

        # Build indices
        self.id_to_img: Dict[int, Dict[str, Any]] = {im["id"]: im for im in self.images}
        self.id_to_catname: Dict[int, str] = {c["id"]: c.get("name", str(c["id"])) for c in self.categories}

        # image_id -> [ann,...]
        self.img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        # Stable order of samples
        self.ids: List[int] = [im["id"] for im in self.images]

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, file_name: str) -> Image.Image:
        img_path = self.image_root / file_name
        if not img_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        info = self.id_to_img[img_id]
        file_name = info["file_name"]

        img = self._load_image(file_name)
        anns = self.img_to_anns.get(img_id, [])

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        crowds: List[int] = []

        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                # skip invalid degenerate boxes
                continue
            boxes.append(_to_xyxy([float(x), float(y), float(w), float(h)]))
            labels.append(int(a["category_id"]))
            areas.append(float(a.get("area", w * h)))
            crowds.append(int(a.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            crowds_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            crowds_tensor = torch.tensor(crowds, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": areas_tensor,
            "iscrowd": crowds_tensor,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"json_path={str(self.json_path)!r}, "
            f"image_root={str(self.image_root)!r}, "
            f"samples={len(self)}"
            ")"
        )


def detection_collate(batch):
    """
    Collate function for detection datasets with variable number of boxes per image.
    Returns:
        images: List[ PIL.Image.Image or Tensor ]
        targets: List[ Dict[str, Tensor] ]
    """
    images, targets = zip(*batch)  # type: ignore
    return list(images), list(targets)
