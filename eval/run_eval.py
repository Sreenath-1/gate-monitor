"""
eval/run_eval.py
================
Evaluation harness for the vehicle ANPR pipeline.

Replays clips with --source, reads labels.csv, and reports metrics:
  - Vehicle detection recall
  - Plate detection recall per vehicle
  - Plate exact-match accuracy
  - Character error rate (CER)
  - Valid-format rate
  - Precision at emit threshold vs coverage
  - Type accuracy
  - FPS, p50/p95 event latency, CPU%, GPU util, VRAM
  - Dropped frames
  - Split by source tag (phone, public, simulated, cctv)

Usage:
    python eval/run_eval.py --source data/clips/test_clip.mp4 --labels data/labels.csv
    python eval/run_eval.py --source data/clips/ --labels data/labels.csv --split dev

Labels CSV format:
    track_uid, plate_text_true, type_true, source, condition
    Example: trk_001, TN09AB1234, car, phone, day
    Conditions: day, night, rain, ir
    Source: phone, public, simulated, cctv
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class LabelRow:
    """A single ground-truth label row."""
    track_uid: str
    plate_text_true: str
    type_true: str
    source: str = "unknown"       # phone | public | simulated | cctv
    condition: str = "day"        # day | night | rain | ir


@dataclass
class PredictionRow:
    """A single prediction from the pipeline."""
    track_uid: str
    plate_text_pred: str
    plate_conf: float
    plate_valid: bool
    vehicle_type_pred: str
    vehicle_detected: bool
    plate_detected: bool
    event_latency_ms: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    # Detection
    vehicle_recall: float = 0.0
    plate_recall_per_vehicle: float = 0.0

    # Plate reading
    plate_exact_match: float = 0.0
    character_error_rate: float = 0.0
    valid_format_rate: float = 0.0

    # Precision / coverage
    precision_at_threshold: float = 0.0
    coverage_at_threshold: float = 0.0
    emit_threshold: float = 0.0

    # Type
    type_accuracy: float = 0.0

    # Performance
    fps: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Resources
    cpu_percent: float = 0.0
    gpu_util_percent: float = 0.0
    vram_mb: float = 0.0
    rss_mb: float = 0.0

    # Reliability
    dropped_frames: int = 0
    total_frames: int = 0
    drop_rate: float = 0.0

    # Counts
    n_labels: int = 0
    n_predictions: int = 0
    n_vehicles_detected: int = 0
    n_plates_detected: int = 0
    n_exact_match: int = 0

    # Splits
    source_tag: str = "all"
    condition_tag: str = "all"


def load_labels(csv_path: str) -> List[LabelRow]:
    """Load ground-truth labels from CSV.

    Expected columns: track_uid, plate_text_true, type_true, [source], [condition]
    """
    labels = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(LabelRow(
                track_uid=row["track_uid"].strip(),
                plate_text_true=row["plate_text_true"].strip().upper(),
                type_true=row["type_true"].strip().lower(),
                source=row.get("source", "unknown").strip().lower(),
                condition=row.get("condition", "day").strip().lower(),
            ))
    return labels


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(pred: str, true: str) -> float:
    """Character Error Rate = edit_distance / len(true)."""
    if not true:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, true) / len(true)


def compute_metrics(
    labels: List[LabelRow],
    predictions: Dict[str, PredictionRow],
    emit_threshold: float = 0.5,
) -> EvalMetrics:
    """Compute evaluation metrics from labels and predictions.

    Args:
        labels: Ground-truth labels.
        predictions: Dict of track_uid -> PredictionRow.
        emit_threshold: Confidence threshold for "emitted" reads.

    Returns:
        EvalMetrics with all fields populated.
    """
    m = EvalMetrics()
    m.n_labels = len(labels)
    m.n_predictions = len(predictions)
    m.emit_threshold = emit_threshold

    if not labels:
        return m

    vehicles_detected = 0
    plates_detected = 0
    exact_matches = 0
    cer_values = []
    valid_format_count = 0
    type_correct = 0
    emitted_correct = 0
    emitted_total = 0
    latencies = []

    for label in labels:
        pred = predictions.get(label.track_uid)
        if pred is None:
            continue

        # Vehicle detection recall
        if pred.vehicle_detected:
            vehicles_detected += 1

        # Plate detection recall (given vehicle was detected)
        if pred.vehicle_detected and pred.plate_detected:
            plates_detected += 1

        # Plate text comparison
        if pred.plate_text_pred and label.plate_text_true:
            pred_text = pred.plate_text_pred.upper().replace(" ", "")
            true_text = label.plate_text_true.upper().replace(" ", "")

            if pred_text == true_text:
                exact_matches += 1

            cer_values.append(character_error_rate(pred_text, true_text))

            if pred.plate_valid:
                valid_format_count += 1

            # Precision at threshold
            if pred.plate_conf >= emit_threshold:
                emitted_total += 1
                if pred_text == true_text:
                    emitted_correct += 1

        # Type accuracy
        if pred.vehicle_type_pred and label.type_true:
            if pred.vehicle_type_pred.lower() == label.type_true.lower():
                type_correct += 1

        if pred.event_latency_ms > 0:
            latencies.append(pred.event_latency_ms)

    # Compute rates
    m.n_vehicles_detected = vehicles_detected
    m.n_plates_detected = plates_detected
    m.n_exact_match = exact_matches

    m.vehicle_recall = vehicles_detected / len(labels) if labels else 0.0
    m.plate_recall_per_vehicle = (
        plates_detected / vehicles_detected if vehicles_detected > 0 else 0.0
    )
    m.plate_exact_match = (
        exact_matches / len(labels) if labels else 0.0
    )
    m.character_error_rate = (
        sum(cer_values) / len(cer_values) if cer_values else 0.0
    )
    m.valid_format_rate = (
        valid_format_count / len(labels) if labels else 0.0
    )
    m.precision_at_threshold = (
        emitted_correct / emitted_total if emitted_total > 0 else 0.0
    )
    m.coverage_at_threshold = (
        emitted_total / len(labels) if labels else 0.0
    )
    m.type_accuracy = type_correct / len(labels) if labels else 0.0

    if latencies:
        latencies.sort()
        m.p50_latency_ms = latencies[len(latencies) // 2]
        m.p95_latency_ms = latencies[int(len(latencies) * 0.95)]

    return m


def compute_split_metrics(
    labels: List[LabelRow],
    predictions: Dict[str, PredictionRow],
    emit_threshold: float = 0.5,
) -> Dict[str, EvalMetrics]:
    """Compute metrics split by source and condition.

    Returns:
        Dict with keys like 'all', 'source:phone', 'condition:night', etc.
    """
    results = {}

    # Overall
    overall = compute_metrics(labels, predictions, emit_threshold)
    overall.source_tag = "all"
    overall.condition_tag = "all"
    results["all"] = overall

    # By source
    sources = set(l.source for l in labels)
    for src in sorted(sources):
        subset = [l for l in labels if l.source == src]
        m = compute_metrics(subset, predictions, emit_threshold)
        m.source_tag = src
        results[f"source:{src}"] = m

    # By condition
    conditions = set(l.condition for l in labels)
    for cond in sorted(conditions):
        subset = [l for l in labels if l.condition == cond]
        m = compute_metrics(subset, predictions, emit_threshold)
        m.condition_tag = cond
        results[f"condition:{cond}"] = m

    return results


def print_report(results: Dict[str, EvalMetrics]):
    """Print a formatted evaluation report."""
    print(f"\n{'='*80}")
    print(f"  VEHICLE ANPR EVALUATION REPORT")
    print(f"{'='*80}")

    for key, m in results.items():
        print(f"\n  ── {key} ({m.n_labels} labels) ─────────────")
        print(f"    Vehicle recall       : {m.vehicle_recall:.1%} "
              f"({m.n_vehicles_detected}/{m.n_labels})")
        print(f"    Plate recall/vehicle : {m.plate_recall_per_vehicle:.1%} "
              f"({m.n_plates_detected}/{m.n_vehicles_detected})")
        print(f"    Plate exact match    : {m.plate_exact_match:.1%} "
              f"({m.n_exact_match}/{m.n_labels})")
        print(f"    Character error rate  : {m.character_error_rate:.3f}")
        print(f"    Valid format rate     : {m.valid_format_rate:.1%}")
        print(f"    Precision@{m.emit_threshold:.2f}        : {m.precision_at_threshold:.1%}")
        print(f"    Coverage@{m.emit_threshold:.2f}         : {m.coverage_at_threshold:.1%}")
        print(f"    Type accuracy        : {m.type_accuracy:.1%}")
        if m.fps > 0:
            print(f"    FPS                  : {m.fps:.1f}")
        if m.p50_latency_ms > 0:
            print(f"    Latency p50/p95      : {m.p50_latency_ms:.0f} / {m.p95_latency_ms:.0f} ms")
        if m.vram_mb > 0:
            print(f"    VRAM                 : {m.vram_mb:.0f} MB")
        if m.rss_mb > 0:
            print(f"    RSS                  : {m.rss_mb:.0f} MB")
        if m.dropped_frames > 0:
            print(f"    Dropped frames       : {m.dropped_frames} ({m.drop_rate:.1%})")

    print(f"\n{'='*80}\n")


def save_report(results: Dict[str, EvalMetrics], output_path: str):
    """Save report as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {k: asdict(v) for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vehicle ANPR pipeline on labelled clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Video file or directory of clips")
    parser.add_argument("--labels", required=True,
                        help="Path to labels.csv")
    parser.add_argument("--split", type=str, default=None,
                        choices=["dev", "test"],
                        help="Evaluate only dev or test split")
    parser.add_argument("--emit-threshold", type=float, default=0.5,
                        help="Confidence threshold for emitted reads")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent.parent / "eval" / "results.json"),
                        help="Output JSON report path")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  EVALUATION HARNESS (skeleton)")
    print(f"{'='*60}")
    print(f"  Source : {args.source}")
    print(f"  Labels : {args.labels}")
    print(f"  Split  : {args.split or 'all'}")
    print()

    # Load labels
    labels = load_labels(args.labels)
    print(f"  Loaded {len(labels)} labels")
    sources = defaultdict(int)
    for l in labels:
        sources[l.source] += 1
    for src, count in sorted(sources.items()):
        print(f"    source:{src} = {count}")

    # TODO: In P1+, this will actually run the pipeline on --source
    # and collect predictions. For now, the skeleton returns empty results.
    print("\n  ⚠ Skeleton mode: no pipeline run. "
          "Returning empty predictions.")
    print("    This will be wired up in P1 (vehicle detect + track).")

    predictions: Dict[str, PredictionRow] = {}

    # Compute metrics
    results = compute_split_metrics(labels, predictions, args.emit_threshold)
    print_report(results)
    save_report(results, args.output)


if __name__ == "__main__":
    main()
