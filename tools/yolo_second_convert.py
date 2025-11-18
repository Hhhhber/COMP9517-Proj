import json
from collections import defaultdict

# ===== 配置 =====
IN_PATH  = "./outputs/deep_yolo/pred_test.json"  # 你现在的检测 json（每条一框）
OUT_PATH = "./outputs/deep_yolo/pred_yolo_test_with_prob_onehot.json"          # 输出的新 json
NUM_CLASSES = 12                                                                  # 类别数
# =================

with open(IN_PATH, "r", encoding="utf-8") as f:
    dets = json.load(f)

# 按 image_id 分组：{image_id: [detections...]}
by_img = defaultdict(list)
for d in dets:
    by_img[d["image_id"]].append(d)

# 对每张图：只保留“全图最强检测框”的分数，其它类别为 0
for img_id, det_list in by_img.items():
    best_cid = None
    best_score = 0.0

    # 找这一张图里 score 最大的那个检测框
    for d in det_list:
        cid = int(d["label"])
        s = float(d["score"])
        if 0 <= cid < NUM_CLASSES and s > best_score:
            best_score = s
            best_cid = cid

    # 构造长度 NUM_CLASSES 的向量：只有 best_cid 这一维是 best_score，其它全 0
    prob_img = [0.0] * NUM_CLASSES
    if best_cid is not None:
        prob_img[best_cid] = best_score

    # 把这一行概率向量挂到该图所有检测框上
    for d in det_list:
        d["prob"] = prob_img

# 展平成一个列表
new_dets = []
for det_list in by_img.values():
    new_dets.extend(det_list)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(new_dets, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_PATH, " we have ", len(new_dets), "detections")
