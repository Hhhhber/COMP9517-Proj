import argparse, json
from pathlib import Path
from .classifier import train_svm, predict_dir

def main():
    parser = argparse.ArgumentParser("Traditional baseline: HOG/LBP + SVM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    #===train===
    p1 = sub.add_parser("train")
    p1.add_argument("--train_dir",  default="./data/agropest12_classic_train")
    p1.add_argument("--feature",    default="hog", choices=["hog", "lbp"])
    p1.add_argument("--model",      default="./outputs/traditional/checkpoints/svm_model.pkl")
    p1.add_argument("--labelmap",   default="./outputs/traditional/labelmap.json")

    #===predict===
    p2 = sub.add_parser("predict")
    #这里给一个更常用的默认 用分类后的 valid 根目录 脚本会 rglob
    p2.add_argument("--images",     default="./data/agropest12_classic_valid")
    p2.add_argument("--feature",    default="hog", choices=["hog", "lbp"])
    p2.add_argument("--model",      default="./outputs/traditional/checkpoints/svm_model.pkl")
    p2.add_argument("--labelmap",   default="./outputs/traditional/labelmap.json")
    p2.add_argument("--out",        default="./outputs/traditional/preds.json")

    args = parser.parse_args()

    #统一创建顶层输出目录
    Path("./outputs/traditional").mkdir(parents=True, exist_ok=True)

    if args.cmd == "train":
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        Path(args.labelmap).parent.mkdir(parents=True, exist_ok=True)
        train_svm(args.train_dir, args.model, args.labelmap, feature=args.feature)
        print(f"!!model saved to {args.model}, label mapping to {args.labelmap}")

    elif args.cmd == "predict":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        preds = predict_dir(args.model, args.labelmap, args.images, feature=args.feature)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"!!predictions={len(preds)} saved to {args.out}")

if __name__ == "__main__":
    main()