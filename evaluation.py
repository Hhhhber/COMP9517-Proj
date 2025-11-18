import json
from pathlib import Path

from metrics_cls import evaluate_classification, plot_confusion_matrix, plot_roc_curve
from metrics_det import evaluate_detection


def main():
    project_root = Path.cwd()

    # 统一的 GT 标签目录（原 YOLO 格式）
    gt_labels = project_root / "data" / "AgroPest-12" / "test" / "labels"

    # 传统方法预测结果（你 run_all 里生成的是这个）
    trad_pred = project_root / "outputs" / "traditional" / "preds_test.json"

    # YOLO / FRCNN 的预测 json（后面 C 同学或你再补生成）
    yolo_pred = project_root / "outputs" / "deep_yolo" / "pred_yolo_test_with_prob_onehot.json"
    frcnn_pred = project_root / "outputs" / "frcnn" / "pred_frcnn_test_with_prob_onehot.json"

    # 评估结果统一放在 outputs/plot_eval/metrics 下面
    metrics_dir = project_root / "outputs" / "plot_eval" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_trad_cls  = metrics_dir / "results_trad_cls.json"
    out_yolo_cls  = metrics_dir / "results_yolo_cls.json"
    out_yolo_det  = metrics_dir / "results_yolo_det.json"
    out_frcnn_cls = metrics_dir / "results_frcnn_cls.json"
    out_frcnn_det = metrics_dir / "results_frcnn_det.json"

    print("=== Traditional SVM ===")
    metrics_trad_cls, y_true_trad, y_pred_trad, y_probs_trad = evaluate_classification(
        gt_label_dir=str(gt_labels),
        pred_path=str(trad_pred),
        output_json=str(out_trad_cls),
        method_name="Traditional-SVM",
        strategy="first",
    )
    plot_confusion_matrix(metrics_trad_cls["confusion_matrix"])
    if y_probs_trad is not None:
        plot_roc_curve(y_true_trad, y_probs_trad)

    print("\n=== YOLO ===")
    metrics_yolo_cls, y_true_yolo, y_pred_yolo, y_probs_yolo = evaluate_classification(
        gt_label_dir=str(gt_labels),
        pred_path=str(yolo_pred),
        output_json=str(out_yolo_cls),
        method_name="YOLO",
        strategy="first",
    )

    metrics_yolo_det = evaluate_detection(
        gt_label_dir=str(gt_labels),
        pred_path=str(yolo_pred),
        output_json=str(out_yolo_det),
        method_name="YOLO",
        num_classes=12,
        iou_thresholds=[0.5],
    )
    plot_confusion_matrix(metrics_yolo_cls["confusion_matrix"])
    if y_probs_yolo is not None:
        plot_roc_curve(y_true_yolo, y_probs_yolo)

    print("\n=== Faster R-CNN ===")
    metrics_frcnn_cls, y_true_frcnn, y_pred_frcnn, y_probs_frcnn = evaluate_classification(
        gt_label_dir=str(gt_labels),
        pred_path=str(frcnn_pred),
        output_json=str(out_frcnn_cls),
        method_name="Faster-RCNN",
        strategy="first",
    )

    metrics_frcnn_det = evaluate_detection(
        gt_label_dir=str(gt_labels),
        pred_path=str(frcnn_pred),
        output_json=str(out_frcnn_det),
        method_name="Faster-RCNN",
        num_classes=12,
        iou_thresholds=[0.5],
    )
    plot_confusion_matrix(metrics_frcnn_cls["confusion_matrix"])
    if y_probs_frcnn is not None:
        plot_roc_curve(y_true_frcnn, y_probs_frcnn)

    # 终端上的总对比打印（保留你同学的风格）
    print("\n" + "=" * 70)
    print("Traditional vs YOLO vs Faster-RCNN")
    print("=" * 70)
    print(
        f"Traditional - Accuracy: {metrics_trad_cls['accuracy']:.2%}, "
        f"F1: {metrics_trad_cls['f1']:.4f}"
    )
    print(
        f"YOLO        - Accuracy: {metrics_yolo_cls['accuracy']:.2%}, "
        f"F1: {metrics_yolo_cls['f1']:.4f}, "
        f"mAP@0.50: {metrics_yolo_det['mAP@0.50']:.4f}"
    )
    print(
        f"Faster-RCNN - Accuracy: {metrics_frcnn_cls['accuracy']:.2%}, "
        f"F1: {metrics_frcnn_cls['f1']:.4f}, "
        f"mAP@0.50: {metrics_frcnn_det['mAP@0.50']:.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
