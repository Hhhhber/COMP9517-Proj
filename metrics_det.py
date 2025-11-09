import numpy as np
import os
import json

def load_ground_truth_boxes(label_dir, img_dir=None):
    """
    格式:
        每行: <class_id> <x_center> <y_center> <width> <height>
        坐标都是归一化的(0-1之间)
        label_dir: 标注文件夹路径
        img_dir: 图像文件夹路径
    
    返回格式:
        gt_annotations: {
            image_name: [
                {'class_id': 0, 'bbox': [x_center, y_center, w, h], 'normalized': True},
                ...
            ]
        }
    """
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
    
    print(f"GT共加载 {len(gt_annotations)} 张图像的检测框")
    total_boxes = sum(len(boxes) for boxes in gt_annotations.values())
    print(f"GT共 {total_boxes} 个目标框, 平均每张 {total_boxes/len(gt_annotations):.2f} 个")
    
    return gt_annotations

def load_prediction_boxes(pred_path):
    """
    支持:
    1. JSON格式: [{"image_name": "xxx.jpg", "class_id": 0, "bbox": [...], "score": 0.95}]
    2. TXT格式: 每行 <class_id> <x_center> <y_center> <width> <height> <confidence>
    
    返回格式:
        pred_annotations: {
            image_name: [
                {'class_id': 0, 'bbox': [x, y, w, h], 'score': 0.95, 'normalized': True},
                ...
            ]
        }
    """
    pred_annotations = {}
    
    if pred_path.endswith('.json'):
        # JSON
        with open(pred_path, 'r') as f:
            data = json.load(f)
        
        for pred in data:
            img_name = pred['image_name']
            class_id = pred['class_id']
            bbox = pred['bbox']
            score = pred.get('score', 1.0)
            
            if img_name not in pred_annotations:
                pred_annotations[img_name] = []
            
            pred_annotations[img_name].append({
                'class_id': class_id,
                'bbox': bbox,
                'score': score,
                'normalized': pred.get('normalized', True)
            })
    
    elif os.path.isdir(pred_path):
        # TXT
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
        raise ValueError(f"不支持格式")
    
    print(f"预测共加载 {len(pred_annotations)} 张图像的预测框")
    total_boxes = sum(len(boxes) for boxes in pred_annotations.values())
    print(f"预测共 {total_boxes} 个预测框, 平均每张 {total_boxes/len(pred_annotations):.2f} 个")
    
    return pred_annotations

def box_iou(box1, box2, format='xywh'):
    """
    计算两个边界框的IoU
    如果bbox格式不同,需要调整
    
    参数:
        box1, box2: [x_center, y_center, width, height] 或 [x1, y1, x2, y2]
        format: 'xywh' (中心点+宽高) 或 'xyxy' (左上+右下)
    
    返回:
        iou: float, 范围 [0, 1]
    """
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
    
    # 计算交集
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # 计算并集
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_ap(precisions, recalls):
    """
    计算单个类别的Average Precision (11点插值法或全精度法)
    当前使用全精度法 (所有recall点)
    """

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_map_per_class(gt_boxes, pred_boxes, class_id, iou_threshold=0.5):
    """
    计算单个类别在某个IoU阈值下的AP
    
    参数:
        gt_boxes: GT标注 (字典)
        pred_boxes: 预测结果 (字典)
        class_id: 类别ID
        iou_threshold: IoU阈值
    
    返回:
        ap: Average Precision
        num_gt: 该类别的GT数量
    """
    # 收集该类别的所有GT和预测
    all_gt = []
    all_pred = []
    
    # 遍历图像
    all_image_names = set(gt_boxes.keys()) | set(pred_boxes.keys())
    
    for img_name in all_image_names:
        # GT
        if img_name in gt_boxes:
            for box in gt_boxes[img_name]:
                if box['class_id'] == class_id:
                    all_gt.append({
                        'image_name': img_name,
                        'bbox': box['bbox'],
                        'detected': False
                    })
        
        # 预测
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
        # 没有GT
        return 0.0, 0
    
    if len(all_pred) == 0:
        # 没有预测
        return 0.0, num_gt

    all_pred = sorted(all_pred, key=lambda x: x['score'], reverse=True)
    
    # 匹配预测和GT
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
    
    # 累积和
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # precision和recall
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # AP
    ap = calculate_ap(precisions, recalls)
    
    return ap, num_gt

def calculate_map(gt_boxes, pred_boxes, num_classes=12, iou_thresholds=[0.5]):
    """
    计算mAP
    
    参数:
        gt_boxes: GT标注 (字典)
        pred_boxes: 预测结果 (字典)
        num_classes: 类别数量
        iou_thresholds: IoU阈值列表 (如 [0.5] 或 [0.5, 0.55, ..., 0.95])
    
    返回:
        results: 字典,包含各种mAP指标
    """
    results = {}
    
    # 每个IoU阈值的mAP
    for iou_thr in iou_thresholds:
        aps = []
        class_aps = {}
        
        for class_id in range(num_classes):
            ap, num_gt = calculate_map_per_class(gt_boxes, pred_boxes, class_id, iou_thr)
            aps.append(ap)
            class_aps[class_id] = {'ap': ap, 'num_gt': num_gt}
     
        # 所有类别的平均mAP
        mAP = np.mean(aps)
        results[f'mAP@{iou_thr:.2f}'] = mAP
        results[f'class_AP@{iou_thr:.2f}'] = class_aps
        
        print(f"[mAP@{iou_thr:.2f}] = {mAP:.4f}")
    
    # 多个IoU阈值,计算平均
    if len(iou_thresholds) > 1:
        avg_mAP = np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])
        results['mAP@0.5:0.95'] = avg_mAP
        print(f"[mAP@0.5:0.95] = {avg_mAP:.4f}")
    
    return results

def print_map_results(results, method_name="模型"):
    """
    打印结果
    """
    print(f"{method_name} - 检测性能评估结果")
    
    for key, value in results.items():
        if key.startswith('mAP@') and not key.endswith(']'):
            print(f"{key:20s}: {value:.4f}")

def save_map_results(results, output_path):
    """
    保存mAP结果到JSON文件
    """
    # 只保存mAP值,不保存每个类别的详细信息
    serializable_results = {
        k: float(v) for k, v in results.items() 
        if k.startswith('mAP@') and isinstance(v, (float, np.floating))
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def evaluate_detection(gt_label_dir, pred_path, output_json=None, method_name="模型", 
                       num_classes=12, iou_thresholds=[0.5]):
    """
    检测评估
    参数:
        gt_label_dir: GT标注文件夹路径
        pred_path: 预测结果路径 (JSON或TXT)
        method_name: 方法名称(用于打印)
        num_classes: 类别数量
        iou_thresholds: IoU阈值列表
    """
    # 1. 加载数据
    gt_boxes = load_ground_truth_boxes(gt_label_dir)
    pred_boxes = load_prediction_boxes(pred_path)
    
    # 2. 计算mAP
    results = calculate_map(gt_boxes, pred_boxes, num_classes, iou_thresholds)
    
    # 3. 打印结果
    print_map_results(results, method_name)
    
    # 4. 保存结果
    if output_json:
        save_map_results(results, output_json)
    
    return results



results = evaluate_detection(
    gt_label_dir='test/labels',
    pred_path='mock_predictions',
    output_json='results_det.json',
    method_name='模拟模型',
    num_classes=12,
    iou_thresholds=[0.5]
)

results_coco = evaluate_detection(
    gt_label_dir='test/labels',
    pred_path='mock_predictions',
    output_json='results_det_coco.json',
    method_name='模拟模型',
    num_classes=12,
    iou_thresholds=[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
)


    
