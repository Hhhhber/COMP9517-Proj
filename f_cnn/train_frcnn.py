# f_cnn/train_frcnn.py — 快速稳定版（AMP/TF32/可选冻结/可换骨干）
import argparse, os, time, json, warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from f_cnn.data import YoloDetectDataset, get_transforms, collate_fn

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# 模型构建
# --------------------------
def build_model(num_classes: int, backbone: str = "resnet50"):
    if backbone == "mobile":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --------------------------
# 主流程
# --------------------------
def main(a):
    # 设备/后端
    device = torch.device("cuda" if torch.cuda.is_available() and a.device == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print(f"\n>> Device: {device}, CUDA available={torch.cuda.is_available()}\n")

    root = Path(a.data_root)

    # 数据集
    train_set = YoloDetectDataset(str(root), "train", a.imgsz, get_transforms(True,  a.imgsz))
    val_set   = YoloDetectDataset(str(root), "valid", a.imgsz, get_transforms(False, a.imgsz))
    num_classes = train_set.num_classes + 1  # + 背景

    # 烟囱测试
    if a.limit and a.limit > 0:
        train_set = Subset(train_set, range(min(a.limit, len(train_set))))
        val_cap   = max(1, a.limit // 5)
        val_set   = Subset(val_set, range(min(val_cap, len(val_set))))

    tkwargs = dict(
        batch_size=a.batch,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if a.workers > 0:
        tkwargs["prefetch_factor"] = 4
        tkwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, **tkwargs)

    # 验证 loader（batch=1 更稳）
    vkwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if a.workers > 0:
        vkwargs["prefetch_factor"] = 4
        vkwargs["persistent_workers"] = True
    val_loader = DataLoader(val_set, **vkwargs)
    # 模型与优化器
    model = build_model(num_classes, backbone=a.backbone).to(device)
    # 仅对“模型”使用 channels_last（输入张量保持 C,H,W）
    model = model.to(memory_format=torch.channels_last)

    # 加载权重
    if a.weights:
        ckpt = torch.load(a.weights, map_location="cpu")
        state = ckpt.get("model", ckpt)   # 兼容 {'model': state_dict} 或直接 state_dict
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f">> loaded weights from {a.weights}")
        if missing:    print("   missing keys:", len(missing))
        if unexpected: print("   unexpected keys:", len(unexpected))

    # RPN/ROI 提案
    model.rpn.pre_nms_top_n_train  = a.rpn_pre_train
    model.rpn.post_nms_top_n_train = a.rpn_post_train
    model.rpn.pre_nms_top_n_test   = a.rpn_pre_test
    model.rpn.post_nms_top_n_test  = a.rpn_post_test
    model.roi_heads.detections_per_img = a.max_dets

    params = [p for p in model.parameters() if p.requires_grad]
    optim  = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.StepLR(optim, step_size=a.lr_step, gamma=a.lr_gamma)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(a.save_dir, exist_ok=True)
    best = float("inf")
    best_path = None
    t0 = time.time()

    eval_every = max(1, a.eval_interval)
    val_limit_batches = a.val_limit if (a.val_limit and a.val_limit > 0) else None

    # 冻结 backbone（前 N 个 epoch）
    def set_backbone_trainable(flag: bool):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = flag
    if a.freeze_backbone_epochs > 0:
        set_backbone_trainable(False)
        print(f">> Freeze backbone for first {a.freeze_backbone_epochs} epochs")

    # --------------------------
    # 训练循环
    # --------------------------
    for epoch in range(1, a.epochs + 1):
        # 解冻时机
        if a.freeze_backbone_epochs > 0 and epoch == a.freeze_backbone_epochs + 1:
            set_backbone_trainable(True)
            print(">> Unfreeze backbone")

        model.train()
        tr_sum, tr_n = 0.0, 0

        for imgs, tgts in train_loader:
            keep_imgs, keep_tgts = [], []
            for im, t in zip(imgs, tgts):
                b = t["boxes"]
                if b.numel() == 0:
                    t["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    t["boxes"] = b.view(-1, 4) if b.ndim == 1 else b[:, :4]
                if t["boxes"].shape[0] > 0:
                    keep_imgs.append(im)
                    keep_tgts.append(t)
            if not keep_imgs:
                continue

            imgs = [im.to(device, non_blocking=True) for im in keep_imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in keep_tgts]

            if tr_n == 0:
                print(">> Sample devices:",
                      imgs[0].device,
                      next(iter(tgts[0].values())).device,
                      next(model.parameters()).device)

            with autocast(enabled=(device.type == "cuda")):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())

            optim.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            tr_sum += loss.item()
            tr_n += 1

        sched.step()
        tr = tr_sum / max(tr_n, 1)

        # --------------------------
        # 验证（抽样 & 间隔全量）
        # --------------------------
        model.eval()
        vl_sum, vl_n = 0.0, 0
        do_full = (epoch % eval_every == 0)
        with torch.no_grad():
            for bi, (imgs, tgts) in enumerate(val_loader):
                keep_imgs, keep_tgts = [], []
                for im, t in zip(imgs, tgts):
                    b = t["boxes"]
                    if b.numel() == 0:
                        t["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    else:
                        t["boxes"] = b.view(-1, 4) if b.ndim == 1 else b[:, :4]
                    if t["boxes"].shape[0] > 0:
                        keep_imgs.append(im)
                        keep_tgts.append(t)
                if not keep_imgs:
                    continue

                imgs = [im.to(device, non_blocking=True) for im in keep_imgs]
                tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in keep_tgts]

                model.train()  # 为拿 loss（torchvision 检测模型在 eval 不返回 loss）
                with autocast(enabled=(device.type == "cuda")):
                    loss_dict = model(imgs, tgts)
                vl_sum += sum(v.item() for v in loss_dict.values())
                vl_n += 1
                model.eval()

                if (not do_full) and (val_limit_batches is not None) and (vl_n >= val_limit_batches):
                    break

        vl = vl_sum / max(vl_n, 1)
        print(f"[{epoch}/{a.epochs}] train={tr:.4f}  val={vl:.4f}")

        # 记录最优
        if vl < best:
            best = vl
            best_path = str(Path(a.save_dir) / "best.pth")
            base_train_ds = getattr(train_set, "dataset", train_set)
            class_names = getattr(base_train_ds, "names", None)
            torch.save(
                {"model": model.state_dict(), "class_names": class_names, "imgsz": a.imgsz},
                best_path,
            )
            print("  >> saved", best_path)

    # 报告
    json.dump(
        {"method": "faster_rcnn_yolo_ann",
         "best_ckpt": best_path,
         "train_time_sec": round(time.time() - t0, 3)},
        open(Path(a.save_dir) / "method_report.json", "w"),
    )
    print("✅ Done. Total time:", round(time.time() - t0, 2), "sec")

# --------------------------
# 参数
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="path to a .pth to finetune from")
    ap.add_argument("--limit", type=int, default=0, help="limit training samples (smoke test)")
    ap.add_argument("--data_root", default="./data/AgroPest-12")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--lr", type=float, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--save_dir", default="./outputs/frcnn")

    # 学习率计划
    ap.add_argument("--lr_step", type=int, default=8)
    ap.add_argument("--lr_gamma", type=float, default=0.1)

    # 验证抽样/间隔
    ap.add_argument("--eval_interval", type=int, default=3, help="validate every N epochs (full)")
    ap.add_argument("--val_limit", type=int, default=50, help="limit val batches per eval (0=full)")

    # RPN/ROI
    ap.add_argument("--rpn_pre_train",  type=int, default=1000)
    ap.add_argument("--rpn_post_train", type=int, default=600)
    ap.add_argument("--rpn_pre_test",   type=int, default=800)
    ap.add_argument("--rpn_post_test",  type=int, default=300)
    ap.add_argument("--max_dets",       type=int, default=100)

    # 冻结与骨干
    ap.add_argument("--freeze_backbone_epochs", type=int, default=3, help="freeze backbone for first N epochs (0=off)")
    ap.add_argument("--backbone", choices=["resnet50", "mobile"], default="resnet50")

    main(ap.parse_args())
