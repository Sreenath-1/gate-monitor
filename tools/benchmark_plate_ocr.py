"""
tools/benchmark_plate_ocr.py
============================
Benchmark the plate detection + OCR pipeline.

Measures:
  - Plate detection: inference time, VRAM usage
  - OCR: inference time per plate crop
  - Combined pipeline: end-to-end per-frame time
  - ONNX EP verification (CUDA vs CPU fallback)

Usage:
    .venv-gpu\\Scripts\\python.exe tools/benchmark_plate_ocr.py
    .venv-gpu\\Scripts\\python.exe tools/benchmark_plate_ocr.py --source data/clips/test.mp4
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import psutil
import torch


def get_gpu_stats():
    """Get GPU temp and VRAM from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {"gpu_temp_c": int(parts[0]), "vram_used_mb": int(parts[1])}
    except Exception:
        return {}


def benchmark_plate_detector(source, n_frames=100):
    """Benchmark the open-image-models plate detector."""
    print(f"\n  {'='*56}")
    print(f"  Plate Detector: yolo-v9-t-384-license-plate-end2end")
    print(f"  {'='*56}")

    torch.cuda.empty_cache()
    vram_before = get_gpu_stats().get("vram_used_mb", 0)

    t_load = time.time()
    try:
        from open_image_models import LicensePlateDetector
        detector = LicensePlateDetector(
            detection_model="yolo-v9-t-384-license-plate-end2end"
        )
    except Exception as e:
        return {"error": f"Plate detector load failed: {e}"}
    load_time = time.time() - t_load

    vram_after = get_gpu_stats().get("vram_used_mb", 0)
    print(f"  Load time: {load_time:.1f}s")
    print(f"  VRAM: {vram_before} -> {vram_after} MB (+{vram_after - vram_before} MB)")

    # Check which EP is being used
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ORT providers: {providers}")
    except Exception:
        pass

    # Open source
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return {"error": f"Cannot open: {source}"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Source: {w}x{h}")

    # Warmup
    for _ in range(3):
        ret, frame = cap.read()
        if ret:
            _ = detector.predict(frame)

    # Benchmark
    times_ms = []
    plates_found = 0

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        results = detector.predict(frame)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000)
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            plates_found += len(results.xyxy)

    cap.release()

    times_arr = np.array(times_ms)
    vram_peak = get_gpu_stats().get("vram_used_mb", 0)

    result = {
        "component": "plate_detector",
        "model": "yolo-v9-t-384-license-plate-end2end",
        "load_time_s": round(load_time, 1),
        "vram_delta_mb": vram_after - vram_before,
        "vram_peak_mb": vram_peak,
        "n_frames": len(times_ms),
        "plates_found": plates_found,
        "inference_ms": {
            "mean": round(float(times_arr.mean()), 1),
            "median": round(float(np.median(times_arr)), 1),
            "p95": round(float(np.percentile(times_arr, 95)), 1),
            "min": round(float(times_arr.min()), 1),
            "max": round(float(times_arr.max()), 1),
        },
        "fps": round(1000 / float(np.median(times_arr)), 1),
    }

    print(f"\n  --- Plate Detector Results ---")
    print(f"  Inference: {result['inference_ms']['median']:.1f}ms median")
    print(f"  FPS: {result['fps']:.1f}")
    print(f"  Plates found: {plates_found} in {len(times_ms)} frames")
    print(f"  VRAM: +{result['vram_delta_mb']} MB")

    del detector
    return result


def benchmark_plate_ocr(source, n_frames=100):
    """Benchmark fast-plate-ocr on plate crops."""
    print(f"\n  {'='*56}")
    print(f"  Plate OCR: fast-plate-ocr (cctv-s-v2-global-model)")
    print(f"  {'='*56}")

    # Load OCR model
    t_load = time.time()
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        ocr = LicensePlateRecognizer("cctv-s-v2-global-model")
    except Exception as e:
        return {"error": f"OCR load failed: {e}"}
    load_time = time.time() - t_load
    print(f"  OCR load time: {load_time:.1f}s")

    # Load plate detector to get crops
    try:
        from open_image_models import LicensePlateDetector
        detector = LicensePlateDetector(
            detection_model="yolo-v9-t-384-license-plate-end2end"
        )
    except Exception as e:
        return {"error": f"Plate detector needed for crops: {e}"}

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return {"error": f"Cannot open: {source}"}

    # Collect plate crops
    print(f"  Collecting plate crops from {n_frames} frames...")
    crops = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        results = detector.predict(frame)
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for box in results.xyxy:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 5 and crop.shape[1] > 5:
                    crops.append(crop)
                    if len(crops) >= 200:
                        break
            if len(crops) >= 200:
                break

    cap.release()
    del detector
    print(f"  Collected {len(crops)} plate crops")

    if not crops:
        # Generate synthetic plate-sized crops for timing
        print(f"  No plates found — generating synthetic 200x60 crops")
        crops = [np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
                 for _ in range(50)]

    # Warmup OCR
    for crop in crops[:3]:
        try:
            _ = ocr.run(crop)
        except Exception:
            pass

    # Benchmark OCR
    times_ms = []
    texts = []
    for crop in crops:
        t0 = time.perf_counter()
        try:
            text = ocr.run(crop)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
            if text:
                texts.append(str(text))
        except Exception:
            pass

    if not times_ms:
        return {"error": "No OCR results"}

    times_arr = np.array(times_ms)

    result = {
        "component": "plate_ocr",
        "model": "cctv-s-v2-global-model",
        "load_time_s": round(load_time, 1),
        "n_crops": len(times_ms),
        "inference_ms": {
            "mean": round(float(times_arr.mean()), 1),
            "median": round(float(np.median(times_arr)), 1),
            "p95": round(float(np.percentile(times_arr, 95)), 1),
            "min": round(float(times_arr.min()), 1),
            "max": round(float(times_arr.max()), 1),
        },
        "fps_equivalent": round(1000 / float(np.median(times_arr)), 1),
        "sample_texts": texts[:10],
    }

    print(f"\n  --- Plate OCR Results ---")
    print(f"  Inference: {result['inference_ms']['median']:.1f}ms median per crop")
    print(f"  Throughput: {result['fps_equivalent']:.0f} crops/sec")
    print(f"  Sample: {texts[:5]}")

    del ocr
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark plate detection + OCR")
    parser.add_argument("--source", type=str, default=None,
                        help="Video file (generates synthetic if not given)")
    parser.add_argument("--n-frames", type=int, default=100)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PLATE DETECTION + OCR BENCHMARK")
    print(f"{'='*60}")

    # Source
    if args.source and os.path.exists(args.source):
        source = args.source
    else:
        # Generate a synthetic video
        print(f"  Generating synthetic test video...")
        tmp = Path(__file__).parent.parent / "data" / "clips" / "_plate_benchmark.mp4"
        os.makedirs(tmp.parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp), fourcc, 25, (1920, 1080))
        for _ in range(args.n_frames + 10):
            frame = np.random.randint(100, 200, (1080, 1920, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        source = str(tmp)

    results = []

    # Plate detector
    r_det = benchmark_plate_detector(source, args.n_frames)
    results.append(r_det)

    # OCR
    r_ocr = benchmark_plate_ocr(source, args.n_frames)
    results.append(r_ocr)

    # Combined budget
    print(f"\n{'='*60}")
    print(f"  COMBINED PIPELINE TIMING BUDGET")
    print(f"{'='*60}")

    if "error" not in r_det and "error" not in r_ocr:
        det_ms = r_det["inference_ms"]["median"]
        ocr_ms = r_ocr["inference_ms"]["median"]
        total_plate_ms = det_ms + ocr_ms
        print(f"  Plate detect : {det_ms:.1f} ms")
        print(f"  Plate OCR    : {ocr_ms:.1f} ms")
        print(f"  Total plate  : {total_plate_ms:.1f} ms")
        print(f"  Budget left for vehicle detect + track @ 12 FPS:")
        frame_budget = 1000 / 12  # ~83 ms
        remaining = frame_budget - total_plate_ms
        print(f"    Frame budget (12 FPS): {frame_budget:.0f} ms")
        print(f"    Plate pipeline:        {total_plate_ms:.1f} ms")
        print(f"    Remaining for detect:  {remaining:.1f} ms")
        if remaining < 20:
            print(f"    WARNING: Very tight! May need async plate pipeline.")
        elif remaining < 40:
            print(f"    OK but tight — async plate pipeline recommended.")
        else:
            print(f"    Comfortable budget.")

    # Save
    out_path = Path(__file__).parent.parent / "eval" / "plate_ocr_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
