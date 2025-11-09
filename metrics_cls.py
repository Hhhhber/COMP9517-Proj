import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import json
import os

def load_ground_truth(label_dir):
    """
    加载Ground Truth
    
    格式:
        每行: <class_id> <x_center> <y_center> <width> <height>
        坐标都是归一化的(0-1之间)
        label_dir: 标注文件夹路径
    
    返回格式:
        image_name_to_labels: {image_name: [class_id, ...]}
        image_name_to_boxes: {image_name: [[x_center, y_center, w, h], ...]}
    """
    image_name_to_labels = {}
    image_name_to_boxes = {}
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        # 提取图像名
        image_name = label_file.replace('.txt', '.jpg')
        
        label_path = os.path.join(label_dir, label_file)
        
        labels = []
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
                    
                    labels.append(class_id)
                    boxes.append([x_center, y_center, width, height])
        
        # 如果图像有多个目标,当前保留所有
        if labels:
            image_name_to_labels[image_name] = labels
            image_name_to_boxes[image_name] = boxes
    
    print(f"GT共加载 {len(image_name_to_labels)} 张图像的标注")
    
    # 统计每张图目标数量
    num_objects_per_image = [len(labels) for labels in image_name_to_labels.values()]
    avg_objects = sum(num_objects_per_image) / len(num_objects_per_image) if num_objects_per_image else 0
    print(f"GT平均每张图 {avg_objects:.2f} 个目标")
    
    # 调试
    if len(image_name_to_labels) == 0:
        print("未加载到任何标注")
    
    return image_name_to_labels, image_name_to_boxes

def load_predictions(pred_path):
    """
    支持:
    1. JSON格式: [{"image_name": "xxx.jpg", "class_id": 3, "score": 0.95, "probs": [...]}]
    2. TXT格式: 每行 <class_id> <x_center> <y_center> <width> <height> <confidence>
    
    返回格式:
        image_name_to_preds: {image_name: [class_id, ...]}
        image_name_to_scores: {image_name: [score, ...]}
        image_name_to_probs: {image_name: [[prob1, prob2, ...], ...]}
    """
    image_name_to_preds = {}
    image_name_to_scores = {}
    image_name_to_probs = {}
    
    # 判断是JSON还是文件夹
    if pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            data = json.load(f)
        
        for pred in data:
            img_name = pred['image_name']
            class_id = pred['class_id']
            score = pred.get('score', 1.0)
            
            if img_name not in image_name_to_preds:
                image_name_to_preds[img_name] = []
                image_name_to_scores[img_name] = []
                image_name_to_probs[img_name] = []
            
            image_name_to_preds[img_name].append(class_id)
            image_name_to_scores[img_name].append(score)
            
            if 'probs' in pred:
                image_name_to_probs[img_name].append(pred['probs'])
        
    elif os.path.isdir(pred_path):
        txt_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            img_name = txt_file.replace('.txt', '.jpg')
            txt_path = os.path.join(pred_path, txt_file)
            
            preds = []
            scores = []
            
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        confidence = float(parts[5])
                        preds.append(class_id)
                        scores.append(confidence)
                    elif len(parts) >= 5:
                        class_id = int(parts[0])
                        preds.append(class_id)
                        scores.append(1.0)
            
            if preds:
                image_name_to_preds[img_name] = preds
                image_name_to_scores[img_name] = scores
    
    else:
        raise ValueError(f"不支持格式")
    
    print(f"预测共加载 {len(image_name_to_preds)} 张图像的预测")
    
    # 统计
    avg_preds = sum(len(p) for p in image_name_to_preds.values()) / len(image_name_to_preds) if image_name_to_preds else 0
    print(f"预测平均每张图 {avg_preds:.2f} 个预测")
    
    # 调试
    if len(image_name_to_probs) > 0:
        print(f"[AUC] {len(image_name_to_probs)} 张图像有概率输出")
    else:
        print("预测中没有概率输出,无法计算AUC")
    
    return image_name_to_preds, image_name_to_scores, image_name_to_probs

def align_gt_and_pred(gt_labels_dict, pred_labels_dict, strategy='first'):
    """
    对齐GT和预测(多目标)
  
    参数:
        gt_labels_dict: {image_name: [class_id, ...]}
        pred_labels_dict: {image_name: [class_id, ...]}
        strategy: 对齐策略
            - 'first': 只取第一个目标
            - 'max_conf': 取置信度最高(需要scores)
            - 'all': 展平所有目标
    
    返回:
        y_true: numpy数组,真实标签
        y_pred: numpy数组,预测标签
        aligned_names: 对齐后的图像名列表
    """
    common_names = sorted(set(gt_labels_dict.keys()) & set(pred_labels_dict.keys()))
    
    y_true_list = []
    y_pred_list = []
    
    for img_name in common_names:
        gt_labels = gt_labels_dict[img_name]
        pred_labels = pred_labels_dict[img_name]
        
        # 策略选择
        if strategy == 'first':
            y_true_list.append(gt_labels[0])
            y_pred_list.append(pred_labels[0] if pred_labels else -1)
        
        elif strategy == 'all':
            min_len = min(len(gt_labels), len(pred_labels))
            y_true_list.extend(gt_labels[:min_len])
            y_pred_list.extend(pred_labels[:min_len])
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # 调试
    missing_in_pred = len(gt_labels_dict) - len(common_names)
    if missing_in_pred > 0:
        print(f"{missing_in_pred} 张GT图像没有对应的预测")
    
    print(f"{len(y_true)} 个样本用于评估 (策略: {strategy})")
    
    # 检查是否有未预测
    num_no_pred = (y_pred == -1).sum()
    if num_no_pred > 0:
        print(f"{num_no_pred} 个样本没有预测")
    
    return y_true, y_pred, common_names

def calculate_classification_metrics(y_true, y_pred, y_probs=None, num_classes=12):
    """
    参数:
        y_true: 真实标签 (numpy数组)
        y_pred: 预测标签 (numpy数组)
        num_classes: 类别数量
    
    返回:
        metrics: 字典,包含所有指标
    """
    
    metrics = {}
    
    # 1. Precision, Recall, F1
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 3. AUC (需要概率)
    if y_probs is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"AUC计算失败: {e}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    # 4. 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # 5. precision/recall
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return metrics

def print_metrics(metrics, method_name="模型"):
    """
    打印
    """
    print(f"{method_name} - 分类性能评估结果")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro):    {metrics['recall']:.4f}")
    print(f"F1 Score (macro):  {metrics['f1']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    
    if metrics['auc'] is not None:
        print(f"AUC (macro):       {metrics['auc']:.4f}")
    else:
        print(f"AUC:               N/A (需要概率输出)")
    
    print(f"{'='*50}\n")

def save_metrics(metrics, output_path):
    """
    保存评估结果到JSON文件
    """
    # 转换numpy类型为Python原生类型
    serializable_metrics = {
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'accuracy': float(metrics['accuracy']),
        'auc': float(metrics['auc']) if metrics['auc'] is not None else None,
        'per_class_precision': metrics['per_class_precision'].tolist(),
        'per_class_recall': metrics['per_class_recall'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

def evaluate_classification(gt_label_dir, pred_path, output_json=None, method_name="模型", strategy='first'):
    """
    分类评估
    
    参数:
        gt_label_dir: Ground Truth标注文件夹路径
        pred_path: 预测结果路径 (JSON或TXT)
        output_json: 结果保存路径
        method_name: 方法名称
        strategy: 多目标对齐策略 (first或all)
    """
    # 1. 加载数据
    gt_labels_dict, gt_boxes_dict = load_ground_truth(gt_label_dir)
    pred_labels_dict, pred_scores_dict, probs_dict = load_predictions(pred_path)
    
    # 2. 对齐数据
    y_true, y_pred, common_names = align_gt_and_pred(gt_labels_dict, pred_labels_dict, strategy)
    
    # 3. 对齐概率输出(如果有)
    y_probs = None
    if len(probs_dict) > 0:
        prob_list = []
        for name in common_names:
            if name in probs_dict and len(probs_dict[name]) > 0:
                prob_list.append(probs_dict[name][0])
            else:
                prob_list.append(None)
        
        # 检查是否所有都有概率
        if all(p is not None for p in prob_list):
            y_probs = np.array(prob_list)
        else:
            print("部分图像缺少概率输出,跳过AUC计算")
    
    # 4. 计算指标
    metrics = calculate_classification_metrics(y_true, y_pred, y_probs)
    
    # 5. 打印结果
    print_metrics(metrics, method_name)
    
    # 6. 保存结果
    if output_json:
        save_metrics(metrics, output_json)
    
    return metrics


metrics = evaluate_classification(
    gt_label_dir='test/labels',
    pred_path='mock_predictions',
    output_json='results_cls.json',
    method_name='模拟模型',
    strategy='first'
)

print(f"准确率: {metrics['accuracy']:.2%}")
print(f"F1分数: {metrics['f1']:.4f}")

import matplotlib.pyplot as plt
cm = metrics['confusion_matrix']
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
