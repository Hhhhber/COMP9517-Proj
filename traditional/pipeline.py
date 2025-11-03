import argparse, json
from pathlib import Path
from .classifier import train_svm, predict_dir

def main():
    parser = argparse.ArgumentParser("Traditional baseline: HOG/LBP + SVM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--train_dir", required=True)
    p1.add_argument("--feature", default="hog", choices=["hog","lbp"])
    p1.add_argument("--model", default="outputs/svm_model.pkl")
    p1.add_argument("--labelmap", default="outputs/labelmap.json")

    p2 = sub.add_parser("predict")
    p2.add_argument("--images", required=True)
    p2.add_argument("--feature", default="hog", choices=["hog","lbp"])
    p2.add_argument("--model", default="outputs/svm_model.pkl")
    p2.add_argument("--labelmap", default="outputs/labelmap.json")
    p2.add_argument("--out", default="outputs/preds_trad.json")

    args = parser.parse_args()
    Path("outputs").mkdir(exist_ok=True, parents=True)

    if args.cmd == "train":
        train_svm(args.train_dir, args.model, args.labelmap, feature=args.feature)
        print(f"!!model have saved to {args.model}, label mapping to {args.labelmap}")

    elif args.cmd == "predict":
        preds = predict_dir(args.model, args.labelmap, args.images, feature=args.feature)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"!!predict {len(preds)} , have saved {args.out}")

if __name__ == "__main__":
    main()
