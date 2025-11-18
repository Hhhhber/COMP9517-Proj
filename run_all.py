import subprocess
import sys
from pathlib import Path
import yaml


# -------------------------
# 小工具
# -------------------------
def run_cmd(cmd_list, cwd=None):
    """打印并执行命令"""
    cmd_str = " ".join(str(x) for x in cmd_list)
    print(f"\n>>> RUN: {cmd_str}")
    subprocess.run(cmd_list, check=True, cwd=cwd)


def load_config(cfg_path="config.yaml"):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# 主流程
# -------------------------
def main():
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "config.yaml")

    # 读取配置
    use_existing = cfg.get("pipeline", {}).get("use_existing", True)
    steps = cfg.get("pipeline", {}).get("steps", {})

    paths = cfg.get("paths", {})
    paths_pre = paths.get("preprocess", {})
    paths_trad = paths.get("traditional", {})
    paths_yolo = paths.get("yolo", {})
    paths_frcnn = paths.get('frcnn', {})
    paths_eval = paths.get("plot_eval", {})

    # 假定源码目录结构
    src_A = project_root / "src" / "A_preprocess"

    # ⭐ 这里改成你现在真实的目录：project_root / "traditional"
    src_B = project_root / "traditional"

    src_C = project_root / "deep_yolo"
    src_F = project_root / 'f_cnn'

    D_eval = project_root / 'evaluation.py'

    # 相应脚本路径
    A_script = src_A / "a_preprocessing.py"

    B_pipeline = src_B / "pipeline.py"
    B_eval_valid = src_B / "eval_valid_from_json.py"   # ⭐ 新增：传统方法的验证评估脚本

    C_train = src_C / "train.py"
    C_predict = src_C / "predict.py"

    F_infer = src_F / 'infer_frcnn.py'
    F_eval = src_F / 'eval_frcnn.py'


    # -------------------------
    # A: 预处理（通常一次性）
    # -------------------------
    if steps.get("preprocess", False):
        if use_existing:
            print("[A] skip preprocess（use_existing=True，assume ./outputs/preprocess exists）")
        else:
            if not A_script.exists():
                print(f"[A] no script：{A_script}，skip A")
            else:
                input_dir = paths_pre.get("input_dir", "./data/AgroPest-12/train/images")
                output_dir = paths_pre.get("output_dir", "./outputs/preprocess")

                # 可选：从 config 读取预处理参数
                pre_cfg = cfg.get("preprocess_params", {})
                s_thresh = pre_cfg.get("s_thresh", 40)
                v_thresh = pre_cfg.get("v_thresh", 50)
                kernel_size = pre_cfg.get("kernel_size", 5)

                run_cmd(
                    [
                        sys.executable,
                        str(A_script),
                        "--input_dir", input_dir,
                        "--output_dir", output_dir,
                        "--s_thresh", str(s_thresh),
                        "--v_thresh", str(v_thresh),
                        "--kernel", str(kernel_size),
                    ],
                    cwd=project_root,
                )

    # -------------------------
    # B: 传统方法（SVM） - 训练
    # -------------------------
    if steps.get("traditional_train", False):
        if use_existing:
            print("[B-train] skip training（use_existing=True，assume ./outputs/traditional/* exist）")
        else:
            # 这里不用再检查 B_pipeline.exists()，因为我们是用 -m 调用
            train_dir = paths_trad.get("classic_train_dir", "./data/agropest12_classic_train")
            ckpt_path = paths_trad.get(
                "ckpt_path",
                "./outputs/traditional/checkpoints/svm_model.pkl",
            )
            labelmap_path = paths_trad.get(
                "labelmap_path",
                "./outputs/traditional/labelmap.json",
            )

            run_cmd(
                [
                    sys.executable,
                    "-m", "traditional.pipeline",
                    "train",
                    "--train_dir", train_dir,
                    "--model", ckpt_path,
                    "--labelmap", labelmap_path,
                ],
                cwd=project_root,
            )
    # -------------------------
    # B: 传统方法（SVM） - 预测 + 验证评估 + 测试预测
    # -------------------------
    if steps.get("traditional_predict", False):
        classic_valid_dir = paths_trad.get(
            "classic_valid_dir",
            "./data/agropest12_classic_valid",
        )
        classic_test_dir = paths_trad.get(
            "classic_test_dir",
            "./data/agropest12_classic_test",
        )

        ckpt_path = paths_trad.get(
            "ckpt_path",
            "./outputs/traditional/checkpoints/svm_model.pkl",
        )
        labelmap_path = paths_trad.get(
            "labelmap_path",
            "./outputs/traditional/labelmap.json",
        )

        preds_valid_json = paths_trad.get(
            "preds_valid_json",
            "./outputs/traditional/preds_valid.json",
        )
        preds_test_json = paths_trad.get(
            "preds_test_json",
            "./outputs/traditional/preds_test.json",
        )
        eval_prefix = paths_trad.get(
            "eval_prefix",
            "./outputs/traditional/eval_valid_",
        )

        # 1) 在验证集上做预测
        print("[B-predict] predict valid set …")
        run_cmd(
            [
                sys.executable,
                "-m", "traditional.pipeline",
                "predict",
                "--images", classic_valid_dir,
                "--model", ckpt_path,
                "--labelmap", labelmap_path,
                "--out", preds_valid_json,
            ],
            cwd=project_root,
        )

        # 2) 对验证集做评估（混淆矩阵、P/R/F1）
        print("[B-predict] evaluate valid set …")
        run_cmd(
            [
                sys.executable,
                "-m", "traditional.eval_valid_from_json",
                "--json", preds_valid_json,
                "--out_prefix", eval_prefix,
            ],
            cwd=project_root,
        )

        # 3) 在测试集上做预测
        print("[B-predict] predict test set …")
        run_cmd(
            [
                sys.executable,
                "-m", "traditional.pipeline",
                "predict",
                "--images", classic_test_dir,
                "--model", ckpt_path,
                "--labelmap", labelmap_path,
                "--out", preds_test_json,
            ],
            cwd=project_root,
        )

    # -------------------------
    # C: 深度学习（YOLO） - 训练
    # -------------------------
    if steps.get("dl_train", False):
        if use_existing:
            print("[C-train] skip YOLO training（use_existing=True，assume best.pt exist）")
        else:
            if not C_train.exists():
                print(f"[C-train] no script：{C_train}, skip C-train")
            else:
                # 目前 train.py 自己内部已经写死 data.yaml 和 project/name
                # 如果你之后改成从 config 读取，再在这里传参数即可
                run_cmd(
                    [sys.executable, str(C_train)],
                    cwd=project_root,
                )

    # -------------------------
    # C: 深度学习（YOLO） - 预测
    # -------------------------
    if steps.get("dl_predict", False):
        if not C_predict.exists():
            print(f"[C-predict] no script：{C_predict}，skip C-predict")
        else:
            # source 使用脚本中的默认值 ./data/AgroPest-12/test/images
            # 如需从 config 里控制，可改成:
            # source = cfg["paths"]["yolo"].get("predict_source", "./data/AgroPest-12/test/images")
            run_cmd(
                [sys.executable, str(C_predict)],
                cwd=project_root,
            )


    # -------------------------
    # C2: Faster R-CNN - 推理
    # -------------------------
    if steps.get("frcnn_infer", False):
        if not F_infer.exists():
            print(f"[C2-infer] no script：{F_infer}，skip FRCNN infer")
        else:
            data_root = paths_frcnn.get("data_root", "./data/AgroPest-12")
            ckpt_path = paths_frcnn.get("ckpt_path", "./outputs/frcnn/best.pth")
            preds_json = paths_frcnn.get("preds_json", "./outputs/frcnn/preds_frcnn_test.json")

            run_cmd(
                [
                    sys.executable,
                    '-m', 'f_cnn.infer_frcnn',
                    "--data_root", data_root,
                    "--ckpt", ckpt_path,
                    "--split", "test",
                    "--out", preds_json,
                    "--device", "cpu",
                    "--workers", "0",
                ],
                cwd=project_root,
            )

    # -------------------------
    # C2: Faster R-CNN - 评估
    # -------------------------
    if steps.get("frcnn_eval", False):
        if not F_eval.exists():
            print(f"[C2-eval] no script：{F_eval}，skip FRCNN eval")
        else:
            data_root = paths_frcnn.get("data_root", "./data/AgroPest-12")
            ckpt_path = paths_frcnn.get("ckpt_path", "./outputs/frcnn/best.pth")
            eval_json = paths_frcnn.get("eval_json", "./outputs/frcnn/eval_frcnn_test.json")

            run_cmd(
                [
                    sys.executable,
                    '-m', 'f_cnn.eval_frcnn',
                    "--data_root", data_root,
                    "--weights", ckpt_path,
                    "--device", "cpu",
                    "--imgsz", "640",
                    "--workers", "0",
                    "--split", "test",
                    "--out", eval_json,
                ],
                cwd=project_root,
            )


    # -------------------------
    # D: 统一评测（分类 + 检测）
    # -------------------------
    if steps.get("evaluate", False):
        if not D_eval.exists():
            print(f"[D] no script：{D_eval}，skip D")
        else:
            run_cmd(
                [
                    sys.executable,
                    str(D_eval),
                ],
                cwd=project_root,
            )

    print("\n=== All Done ===")


if __name__ == "__main__":
    main()

