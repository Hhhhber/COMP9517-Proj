import numpy as np
import os
import json


def normalize_image_name(image_name):
    if not image_name.endswith('.jpg'):
        return image_name + '.jpg'
    return image_name


def load_ground_truth_boxes(label_dir, img_dir=None):
    gt_annotations = {}

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        image_name = label_file.replace('.txt', '.jpg')
        label_path = os.path.join(label_dir, label_file)

        boxes = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],
                        'normalized': True
                    })

        if boxes:
            gt_annotations[image_name] = boxes
    total_boxes = sum(len(boxes) for boxes in gt_annotations.values())

    return gt_annotations


def load_prediction_boxes(pred_path, img_size=640, auto_fix_label_offset=True):
    pred_annotations = {}

    if pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            data = json.load(f)

        if len(data) > 0:
            sample = data[0]

            if 'image_id' in sample and 'label' in sample and 'box' in sample:
                all_labels = [item['label'] for item in data]
                min_label = min(all_labels)
                max_label = max(all_labels)

                label_offset = 0
                if auto_fix_label_offset and min_label >= 1:
                    label_offset = -1

                for pred in data:
                    img_name = normalize_image_name(pred['image_id'])
                    class_id = pred['label'] + label_offset
                    score = pred.get('score', 1.0)
                    box = pred['box']

                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2 / img_size
                    y_center = (y1 + y2) / 2 / img_size
                    width = (x2 - x1) / img_size
                    height = (y2 - y1) / img_size

                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    if img_name not in pred_annotations:
                        pred_annotations[img_name] = []

                    pred_annotations[img_name].append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],
                        'score': score,
                        'normalized': True
                    })

            else:
                raise ValueError("unrecognised json format")

    elif os.path.isdir(pred_path):

        txt_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]

        for txt_file in txt_files:
            img_name = txt_file.replace('.txt', '.jpg')
            txt_path = os.path.join(pred_path, txt_file)

            boxes = []

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])

                        boxes.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height],
                            'score': confidence,
                            'normalized': True
                        })

            if boxes:
                pred_annotations[img_name] = boxes

    else:
        raise ValueError(f"unsupported json format: {pred_path}")
    total_boxes = sum(len(boxes) for boxes in pred_annotations.values())

    return pred_annotations


def box_iou(box1, box2, format='xywh'):
    if format == 'xywh':
        x1_1 = box1[0] - box1[2] / 2
        y1_1 = box1[1] - box1[3] / 2
        x2_1 = box1[0] + box1[2] / 2
        y2_1 = box1[1] + box1[3] / 2

        x1_2 = box2[0] - box2[2] / 2
        y1_2 = box2[1] - box2[3] / 2
        x2_2 = box2[0] + box2[2] / 2
        y2_2 = box2[1] + box2[3] / 2
    else:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def calculate_ap(precisions, recalls):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def calculate_map_per_class(gt_boxes, pred_boxes, class_id, iou_threshold=0.5):
    all_gt = []
    all_pred = []

    all_image_names = set(gt_boxes.keys()) | set(pred_boxes.keys())

    for img_name in all_image_names:
        if img_name in gt_boxes:
            for box in gt_boxes[img_name]:
                if box['class_id'] == class_id:
                    all_gt.append({
                        'image_name': img_name,
                        'bbox': box['bbox'],
                        'detected': False
                    })

        if img_name in pred_boxes:
            for box in pred_boxes[img_name]:
                if box['class_id'] == class_id:
                    all_pred.append({
                        'image_name': img_name,
                        'bbox': box['bbox'],
                        'score': box['score']
                    })

    num_gt = len(all_gt)

    if num_gt == 0:
        return 0.0, 0

    if len(all_pred) == 0:
        return 0.0, num_gt

    all_pred = sorted(all_pred, key=lambda x: x['score'], reverse=True)

    tp = np.zeros(len(all_pred))
    fp = np.zeros(len(all_pred))

    for i, pred in enumerate(all_pred):
        img_gt = [gt for gt in all_gt if gt['image_name'] == pred['image_name']]

        if len(img_gt) == 0:
            fp[i] = 1
            continue

        ious = [box_iou(pred['bbox'], gt['bbox'], format='xywh') for gt in img_gt]
        max_iou = max(ious)
        max_idx = ious.index(max_iou)

        if max_iou >= iou_threshold:
            if not img_gt[max_idx]['detected']:
                tp[i] = 1
                img_gt[max_idx]['detected'] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = calculate_ap(precisions, recalls)

    return ap, num_gt


def calculate_map(gt_boxes, pred_boxes, num_classes=12, iou_thresholds=[0.5]):
    results = {}

    for iou_thr in iou_thresholds:
        aps = []
        class_aps = {}

        for class_id in range(num_classes):
            ap, num_gt = calculate_map_per_class(gt_boxes, pred_boxes, class_id, iou_thr)
            aps.append(ap)
            class_aps[class_id] = {'ap': ap, 'num_gt': num_gt}

        mAP = np.mean(aps)
        results[f'mAP@{iou_thr:.2f}'] = mAP
        results[f'class_AP@{iou_thr:.2f}'] = class_aps

        print(f"[mAP@{iou_thr:.2f}] = {mAP:.4f}")

    if len(iou_thresholds) > 1:
        avg_mAP = np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])
        results['mAP@0.5:0.95'] = avg_mAP
        print(f"[mAP@0.5:0.95] = {avg_mAP:.4f}")

    return results


def print_map_results(results, method_name="模型"):
    print(f"{method_name} - detection performance")

    for key, value in results.items():
        if key.startswith('mAP@') and not key.endswith(']'):
            print(f"{key:20s}: {value:.4f}")


def save_map_results(results, output_path):
    serializable_results = {
        k: float(v) for k, v in results.items()
        if k.startswith('mAP@') and isinstance(v, (float, np.floating))
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def evaluate_detection(gt_label_dir, pred_path, output_json=None, method_name="模型",
                       num_classes=12, iou_thresholds=[0.5]):
    gt_boxes = load_ground_truth_boxes(gt_label_dir)
    pred_boxes = load_prediction_boxes(pred_path)

    results = calculate_map(gt_boxes, pred_boxes, num_classes, iou_thresholds)

    print_map_results(results, method_name)

    if output_json:
        save_map_results(results, output_json)

    return results
