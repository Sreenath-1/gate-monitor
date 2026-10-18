"""
tools/eval_mask_classifier.py
=============================
Evaluate mask classifier accuracy on a labeled dataset.

Generates a confusion matrix + per-class metrics to diagnose
the suspected false-positive bias in the MobileNetV2 mask classifier.

Usage:
    python tools/eval_mask_classifier.py --data-dir labeled_faces/
    
    labeled_faces/ structure:
      masked/       ← images of faces wearing masks
      unmasked/     ← images of faces not wearing masks

    Output: confusion matrix, precision/recall/F1, FPR analysis.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def evaluate(data_dir: str, threshold: float = 0.5):
    """Run mask classifier on labeled dataset and report metrics."""
    import onnxruntime as ort

    mask_model = os.path.join(os.path.dirname(__file__), "..",
                               "weights", "mask_classifier.onnx")
    if not os.path.exists(mask_model):
        print(f"ERROR: {mask_model} not found")
        return

    sess = ort.InferenceSession(mask_model, providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    results = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    all_scores = []

    for label_dir, true_label in [("masked", True), ("unmasked", False)]:
        dir_path = os.path.join(data_dir, label_dir)
        if not os.path.isdir(dir_path):
            print(f"  Skipping {dir_path} (not found)")
            continue

        files = [f for f in os.listdir(dir_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {label_dir}: {len(files)} images")

        for fname in files:
            img = cv2.imread(os.path.join(dir_path, fname))
            if img is None:
                continue

            # Preprocess
            img_resized = cv2.resize(img, (112, 112))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            inp = img_rgb.astype(np.float32) / 255.0
            inp = (inp - mean) / std
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, axis=0)

            ort_inputs = {sess.get_inputs()[0].name: inp}
            ort_outs = sess.run(None, ort_inputs)
            logit = ort_outs[0][0][0]
            prob = 1.0 / (1.0 + np.exp(-logit))
            pred_masked = prob > threshold

            all_scores.append((true_label, prob, fname))

            if true_label and pred_masked:
                results["TP"] += 1
            elif true_label and not pred_masked:
                results["FN"] += 1
            elif not true_label and pred_masked:
                results["FP"] += 1
            else:
                results["TN"] += 1

    # Report
    tp, fp, tn, fn = results["TP"], results["FP"], results["TN"], results["FN"]
    total = tp + fp + tn + fn

    print(f"\n{'='*50}")
    print(f"  MASK CLASSIFIER EVALUATION (threshold={threshold})")
    print(f"{'='*50}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Masked  Unmasked")
    print(f"  Actual Masked  {tp:5d}   {fn:5d}")
    print(f"  Actual Unmasked{fp:5d}   {tn:5d}")

    if total > 0:
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n  Accuracy  : {accuracy:.3f}")
        print(f"  Precision : {precision:.3f}")
        print(f"  Recall    : {recall:.3f}")
        print(f"  F1 Score  : {f1:.3f}")
        print(f"  FPR       : {fpr:.3f}  ← false positive rate on unmasked faces")

        if fpr > 0.3:
            print(f"\n  ⚠ HIGH FALSE POSITIVE RATE ({fpr:.1%})")
            print(f"    The classifier flags unmasked faces as masked too often.")
            print(f"    Recommendation: treat mask_state as a feature, not a gate.")

    # Score distribution
    masked_scores = [s[1] for s in all_scores if s[0]]
    unmasked_scores = [s[1] for s in all_scores if not s[0]]

    if masked_scores:
        print(f"\n  Masked scores  : mean={np.mean(masked_scores):.3f} "
              f"std={np.std(masked_scores):.3f}")
    if unmasked_scores:
        print(f"  Unmasked scores: mean={np.mean(unmasked_scores):.3f} "
              f"std={np.std(unmasked_scores):.3f}")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mask classifier on labeled dataset")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with masked/ and unmasked/ subdirs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()
    evaluate(args.data_dir, args.threshold)


if __name__ == "__main__":
    main()
