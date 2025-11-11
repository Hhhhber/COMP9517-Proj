import argparse, glob, os, time
from pathlib import Path 
from datetime import datetime
from ultralytics import YOLO

RUNS_ROOT = "/outputs/deep_learning/detect"

def latest_best(runs_root=RUNS_ROOT):
    trains = sorted(glob.glob(os.path.join(runs_root, "train*")), key=os.path.getmtime)
    for d in reversed(trains):
        cand = os.path.join(d, "weights", "best.pt")
        if os.path.isfile(cand):
            return cand
    return None

def main():
    parser = argparse.ArgumentParser(description="YOLO predict helper")
    parser.add_argument("--weights", type=str, default=None, help="path to best.pt")
    parser.add_argument("--source",  type=str, default="/data/AgroPest-12/test/images", help="images or folder")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--save_txt", action="store_true", help="save detections to TXT")
    args = parser.parse_args()
    Path(RUNS_ROOT).mkdir(parents=True, exist_ok=True)
    weights = args.weights or latest_best()
    if not weights:
        raise FileNotFoundError("Could not find best.pt under runs/detect/train*/weights/. Use --weights to specify.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name  = f"predict_{stamp}"

    print(f"[INFO] Using weights: {weights}")
    print(f"[INFO] Source:        {args.source}")
    print(f"[INFO] Saving to:     {os.path.join(RUNS_ROOT, name)}")

    model = YOLO(weights)
    model.predict(
        source=args.source,
        save=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_txt=args.save_txt,
        project=RUNS_ROOT,
        name=name
    )
    print("[DONE] Predictions saved.")
    print(f"      Folder: {os.path.join(RUNS_ROOT, name)}")

if __name__ == "__main__":
    main()
