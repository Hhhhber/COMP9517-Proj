# COMP9517-Proj
项目根/
├─ data/
│  └─ AgroPest-12/            # Kaggle 原始数据（含 data.yaml / train / valid / test）
│
├─ traditional/
│  ├─ prepare_dataset.py
│  ├─ features.py
│  ├─ classifier.py
│  ├─ pipeline.py
│  └─ eval_valid_from_json.py
└─ outputs/                   # 运行后自动生成

1) 准备数据集（从 YOLO → 分类目录）
需要 改 3 次代码 + 运行 3 次脚本 来分别生成 train / valid / test 三套分类目录。

打开 traditional/prepare_dataset.py，定位到这三行（示例）：

YAML_PATH = Path("./data/AgroPest-12/data.yaml")  # 不变
cfg  = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
root = YAML_PATH.parent

# 下面三行需要根据 需要（train 或者 val 或者 test） 修改：
train_img = (root / cfg["..."]).resolve()               # ... ← train / val / test 三选一
train_lbl = (root / cfg["..."].replace("images","labels")).resolve()
out_root  = Path("data/agropest12_classic_...").resolve()  # 输出目录：train/valid/test

命令行运行(每次改完代码都得运行一次)：
python -m traditional.prepare_dataset

生成完成后的期望结构：
data/
├─ AgroPest-12/                 # 原始 YOLO
├─ agropest12_classic_train/    # 分类版 train
├─ agropest12_classic_valid/    # 分类版 valid
└─ agropest12_classic_test/     # 分类版 test

2) 训练（HOG + SVM）
训练集路径按上一步生成的 data/agropest12_classic_train/

默认（快速）版：每类最多 200 张（开发调试用）（如果没改过 classifier.py，通常就是这个设置）
python -m traditional.pipeline train --train_dir data/agropest12_classic_train

（全量）版：取消限额，提取全量特征再训练
打开 traditional/classifier.py，找到训练数据加载这一行：
X, y, name2id, id2name = load_features(train_dir, feature, limit_per_class=200)

改为（不限制数量）：
X, y, name2id, id2name = load_features(train_dir, feature, limit_per_class=0)

保存后运行：
python -m traditional.pipeline train --train_dir data/agropest12_classic_train

训练输出（默认保存到）：
outputs/traditional/checkpoints/svm_model.pkl
outputs/traditional/labelmap.json

3) 验证集预测 & 评估
3.1 验证集预测（递归扫描 12 类文件夹）
python -m traditional.pipeline predict --images data/agropest12_classic_valid --out outputs/traditional/preds_valid.json

3.2 验证集评估（Accuracy / P/R/F1 / 混淆矩阵）
python -m traditional.eval_valid_from_json --json outputs/traditional/preds_valid.json --out_prefix outputs/traditional/eval_valid_
输出：
outputs/traditional/eval_valid_eval_report.csv
outputs/traditional/eval_valid_confusion_matrix.csv
outputs/traditional/eval_valid_confusion_matrix.png
outputs/traditional/eval_valid_predictions_with_truth.csv

4) 测试集预测（用于最终结果/融合）
python -m traditional.pipeline predict --images data/agropest12_classic_test --out outputs/traditional/preds_test.json


