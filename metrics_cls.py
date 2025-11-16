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


def normalize_image_name(image_name):
    """
    处理.jpg后缀
    """
    if not image_name.endswith('.jpg'):
        return image_name + '.jpg'
    return image_name


def load_ground_truth(label_dir):
    """
    加载Ground Truth标注 - YOLO TXT格式
    """
    image_name_to_labels = {}
    image_name_to_boxes = {}
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
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
        
        if labels:
            image_name_to_labels[image_name] = labels
            image_name_to_boxes[image_name] = boxes
    
    num_objects_per_image = [len(labels) for labels in image_name_to_labels.values()]
    avg_objects = sum(num_objects_per_image) / len(num_objects_per_image) if num_objects_per_image else 0
    
    return image_name_to_labels, image_name_to_boxes


def load_predictions(pred_path, img_size=640, auto_fix_label_offset=True):
    """
    加载模型预测结果
    处理缺少.jpg后缀的image_id, 检测并修正类别ID偏移(1-12 → 0-11)
    """
    image_name_to_preds = {}
    image_name_to_scores = {}
    image_name_to_probs = {}
    
    if pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > 0:
            sample = data[0]
            
            if 'image_id' in sample and 'label' in sample:

                all_labels = [item['label'] for item in data]
                min_label = min(all_labels)
                max_label = max(all_labels)
                
                label_offset = 0
                if auto_fix_label_offset and min_label >= 1:
                    label_offset = -1
                
                for pred in data:
                    img_name = normalize_image_name(pred['image_id'])
                    
                    # 修正类别ID
                    class_id = pred['label'] + label_offset
                    score = pred.get('score', 1.0)
                    
                    if img_name not in image_name_to_preds:
                        image_name_to_preds[img_name] = []
                        image_name_to_scores[img_name] = []
                        image_name_to_probs[img_name] = []
                    
                    image_name_to_preds[img_name].append(class_id)
                    image_name_to_scores[img_name].append(score)
                    
                    if 'probs' in pred:
                        image_name_to_probs[img_name].append(pred['probs'])
            
            else:
                raise ValueError("无法识别的JSON格式")
    
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
        raise ValueError(f"❌ 不支持的预测格式: {pred_path}")
    
    avg_preds = sum(len(p) for p in image_name_to_preds.values()) / len(image_name_to_preds) if image_name_to_preds else 0
    
    return image_name_to_preds, image_name_to_scores, image_name_to_probs


def align_gt_and_pred(gt_labels_dict, pred_labels_dict, strategy='first'):
    """
    对齐GT和预测
    """
    common_names = sorted(set(gt_labels_dict.keys()) & set(pred_labels_dict.keys()))
    
    if len(common_names) == 0:
        print("\n GT和预测没有共同的图像!")
        print(f"GT示例: {sorted(gt_labels_dict.keys())[:3]}")
        print(f"预测示例: {sorted(pred_labels_dict.keys())[:3]}")
        raise ValueError("GT和预测没有共同的图像")
    
    y_true_list = []
    y_pred_list = []
    
    for img_name in common_names:
        gt_labels = gt_labels_dict[img_name]
        pred_labels = pred_labels_dict[img_name]
        
        if strategy == 'first':
            y_true_list.append(gt_labels[0])
            y_pred_list.append(pred_labels[0] if pred_labels else -1)
        
        elif strategy == 'all':
            min_len = min(len(gt_labels), len(pred_labels))
            y_true_list.extend(gt_labels[:min_len])
            y_pred_list.extend(pred_labels[:min_len])
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    missing_in_pred = len(gt_labels_dict) - len(common_names)
    if missing_in_pred > 0:
        print(f"{missing_in_pred} 张GT图像没有对应的预测")
    
    print(f"{len(y_true)} 个样本用于评估 (策略: {strategy})")
    
    num_no_pred = (y_pred == -1).sum()
    if num_no_pred > 0:
        print(f"{num_no_pred} 个样本没有预测")
    
    return y_true, y_pred, common_names


def calculate_classification_metrics(y_true, y_pred, y_probs=None, num_classes=12):
    """
    计算分类指标
    """
    metrics = {}
    
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if y_probs is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return metrics


def print_metrics(metrics, method_name="模型"):
    """
    打印评估结果
    """
    print(f"{method_name} - 分类性能评估")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro):    {metrics['recall']:.4f}")
    print(f"F1 Score (macro):  {metrics['f1']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    
    if metrics['auc'] is not None:
        print(f"AUC (macro):       {metrics['auc']:.4f}")
    
    print("="*50)


def save_metrics(metrics, output_path):
    """
    保存评估结果
    """
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
    完整的分类评估流程
    """
    gt_labels_dict, gt_boxes_dict = load_ground_truth(gt_label_dir)
    pred_labels_dict, pred_scores_dict, probs_dict = load_predictions(pred_path)
    
    y_true, y_pred, common_names = align_gt_and_pred(gt_labels_dict, pred_labels_dict, strategy)
    
    y_probs = None
    if len(probs_dict) > 0:
        prob_list = []
        for name in common_names:
            if name in probs_dict and len(probs_dict[name]) > 0:
                prob_list.append(probs_dict[name][0])
            else:
                prob_list.append(None)
        
        if all(p is not None for p in prob_list):
            y_probs = np.array(prob_list)
    
    metrics = calculate_classification_metrics(y_true, y_pred, y_probs)
    
    print_metrics(metrics, method_name)
    
    if output_json:
        save_metrics(metrics, output_json)
    
    return metrics

'''
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
'''