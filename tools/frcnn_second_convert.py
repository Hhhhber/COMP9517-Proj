import json
from collections import defaultdict

def add_prob(IN_PATH, OUT_PATH, NUM_CLASSES=12):
    # 读取原始检测结果
    with open(IN_PATH, "r", encoding="utf-8") as f:
        dets = json.load(f)

    # 按 image_id 分组
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)

    # 对每一张图，找分数最高的那个类别，做 one-hot 概率向量
    for img_id, det_list in by_img.items():
        best_cid = None
        best_score = 0.0

        for d in det_list:
            cid = int(d["label"])
            s = float(d["score"])
            if 0 <= cid < NUM_CLASSES and s > best_score:
                best_score = s
                best_cid = cid

        prob_img = [0.0] * NUM_CLASSES
        if best_cid is not None:
            prob_img[best_cid] = best_score

        # 给这一张图的所有检测框都挂上同一行 prob 向量
        for d in det_list:
            d["prob"] = prob_img

    # 展平成列表
    new_dets = []
    for det_list in by_img.values():
        new_dets.extend(det_list)

    # 保存
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_dets, f, ensure_ascii=False, indent=2)

    print("Saved:", OUT_PATH, " we have ", len(new_dets), "detections")


if __name__ == "__main__":
    IN_PATH  = "./outputs/frcnn/frcnn_test_middle.json"
    OUT_PATH = "./outputs/frcnn/pred_frcnn_test_with_prob_onehot.json"
    NUM_CLASSES = 12

    add_prob(IN_PATH, OUT_PATH, NUM_CLASSES)
