"""
tools/benchmark_rfdetr.py
=========================
Benchmark RF-DETR Nano vs Small on this hardware.

Measures:
  - VRAM usage (MB)
  - Inference time per frame (ms)
  - Sustained FPS over N frames
  - GPU temperature + throttle status
  - Vehicle detection count and confidence distribution
  - RSS memory

Usage:
    .venv-gpu\\Scripts\\python.exe tools/benchmark_rfdetr.py
    .venv-gpu\\Scripts\\python.exe tools/benchmark_rfdetr.py --source data/clips/test.mp4
    .venv-gpu\\Scripts\\python.exe tools/benchmark_rfdetr.py --fp16
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import psutil
import torch

# COCO class IDs for vehicles
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def get_gpu_stats():
    """Get GPU temp, power, clocks from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.gr,"
             "clocks_throttle_reasons.active,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_temp_c": _safe_int(parts[0]),
            "power_w": _safe_float(parts[1]),
            "clock_mhz": _safe_int(parts[2]),
            "throttle": parts[3] if len(parts) > 3 else None,
            "vram_used_mb": _safe_int(parts[4]) if len(parts) > 4 else None,
        }
    except Exception:
        return {}


def _safe_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def benchmark_model(model_class, model_name, source, n_frames=200, threshold=0.35):
    """Benchmark a single RF-DETR model variant.

    Args:
        model_class: RF-DETR model class (e.g. RFDETRBase).
        model_name: Human-readable name for reporting.
        source: Video file path or 0 for webcam.
        n_frames: Number of frames to benchmark.
        threshold: Detection confidence threshold.

    Returns:
        Dict with benchmark results.
    """
    print(f"\n  {'='*56}")
    print(f"  Benchmarking: {model_name}")
    print(f"  {'='*56}")

    # Pre-benchmark GPU state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1)
    vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    gpu_before = get_gpu_stats()

    print(f"  VRAM before model load: {vram_before:.0f} MB")
    print(f"  GPU temp before: {gpu_before.get('gpu_temp_c', '?')}C")

    # Load model
    t_load_start = time.time()
    try:
        model = model_class()
    except Exception as e:
        return {"model": model_name, "error": f"Load failed: {e}"}
    t_load = time.time() - t_load_start

    vram_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_peak_load = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"  Model loaded in {t_load:.1f}s")
    print(f"  VRAM after load: {vram_after_load:.0f} MB (peak: {vram_peak_load:.0f} MB)")

    # Open video source
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        return {"model": model_name, "error": f"Cannot open source: {source}"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Source: {w}x{h} @ {src_fps:.0f} FPS")

    # Warmup (3 frames)
    print(f"  Warmup...")
    for _ in range(3):
        ret, frame = cap.read()
        if ret:
            _ = model.predict(frame, threshold=threshold)
    torch.cuda.synchronize()

    # Reset stats after warmup
    torch.cuda.reset_peak_memory_stats()
    vram_after_warmup = torch.cuda.memory_allocated() / (1024 * 1024)

    # Benchmark loop
    times_ms = []
    vehicle_counts = []
    confidences = []
    gpu_temps = []
    process = psutil.Process()

    print(f"  Running {n_frames} frames...")
    t_benchmark_start = time.time()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            # Loop back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        detections = model.predict(frame, threshold=threshold)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        times_ms.append(elapsed_ms)

        # Count vehicles
        n_vehicles = 0
        if hasattr(detections, 'class_id') and detections.class_id is not None:
            for cls_id, conf in zip(detections.class_id, detections.confidence):
                if int(cls_id) in VEHICLE_CLASS_IDS:
                    n_vehicles += 1
                    confidences.append(float(conf))
        vehicle_counts.append(n_vehicles)

        # Sample GPU stats every 20 frames
        if i % 20 == 0:
            gs = get_gpu_stats()
            if gs.get("gpu_temp_c"):
                gpu_temps.append(gs["gpu_temp_c"])
            if (i + 1) % 50 == 0:
                fps_so_far = (i + 1) / (time.time() - t_benchmark_start)
                print(f"    Frame {i+1}/{n_frames}: "
                      f"{elapsed_ms:.1f}ms, {fps_so_far:.1f} FPS, "
                      f"temp={gs.get('gpu_temp_c', '?')}C, "
                      f"vehicles={n_vehicles}")

    t_benchmark_total = time.time() - t_benchmark_start
    cap.release()

    # Post-benchmark stats
    vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_current = torch.cuda.memory_allocated() / (1024 * 1024)
    gpu_after = get_gpu_stats()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    # Compute metrics
    times_arr = np.array(times_ms)
    fps_sustained = len(times_ms) / t_benchmark_total if t_benchmark_total > 0 else 0

    result = {
        "model": model_name,
        "source_resolution": f"{w}x{h}",
        "n_frames": len(times_ms),
        "load_time_s": round(t_load, 1),
        "vram_model_mb": round(vram_after_load, 0),
        "vram_peak_mb": round(vram_peak, 0),
        "vram_current_mb": round(vram_current, 0),
        "rss_mb": round(rss_mb, 0),
        "inference_ms": {
            "mean": round(float(times_arr.mean()), 1),
            "median": round(float(np.median(times_arr)), 1),
            "p95": round(float(np.percentile(times_arr, 95)), 1),
            "p99": round(float(np.percentile(times_arr, 99)), 1),
            "min": round(float(times_arr.min()), 1),
            "max": round(float(times_arr.max()), 1),
        },
        "fps_sustained": round(fps_sustained, 1),
        "fps_theoretical": round(1000 / float(np.median(times_arr)), 1) if len(times_arr) > 0 else 0,
        "vehicles_detected_total": sum(vehicle_counts),
        "vehicles_per_frame_mean": round(np.mean(vehicle_counts), 2) if vehicle_counts else 0,
        "confidence_mean": round(np.mean(confidences), 3) if confidences else 0,
        "confidence_median": round(np.median(confidences), 3) if confidences else 0,
        "gpu_temp_start_c": gpu_before.get("gpu_temp_c"),
        "gpu_temp_end_c": gpu_after.get("gpu_temp_c"),
        "gpu_temp_max_c": max(gpu_temps) if gpu_temps else None,
        "throttled": any(t != "0x0000000000000000" and t != "0x0000000000000001"
                        for t in [gpu_after.get("throttle", "")] if t),
    }

    # Print summary
    print(f"\n  --- {model_name} Results ---")
    print(f"  VRAM model : {result['vram_model_mb']:.0f} MB")
    print(f"  VRAM peak  : {result['vram_peak_mb']:.0f} MB")
    print(f"  RSS        : {result['rss_mb']:.0f} MB")
    print(f"  Inference  : {result['inference_ms']['median']:.1f}ms median, "
          f"{result['inference_ms']['p95']:.1f}ms p95")
    print(f"  FPS        : {result['fps_sustained']:.1f} sustained, "
          f"{result['fps_theoretical']:.1f} theoretical")
    print(f"  Vehicles   : {result['vehicles_detected_total']} total, "
          f"{result['vehicles_per_frame_mean']:.1f}/frame")
    print(f"  GPU temp   : {result['gpu_temp_start_c']}C -> {result['gpu_temp_end_c']}C "
          f"(max {result['gpu_temp_max_c']}C)")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    time.sleep(2)

    return result


def generate_test_frames(n=200, w=1920, h=1080):
    """Generate synthetic test frames if no video source available."""
    print(f"  Generating {n} synthetic {w}x{h} test frames...")
    tmp_path = Path(__file__).parent.parent / "data" / "clips" / "_benchmark_synthetic.mp4"
    os.makedirs(tmp_path.parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 25, (w, h))

    for i in range(n):
        frame = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)
        # Add some rectangular shapes to simulate vehicles
        for _ in range(np.random.randint(0, 4)):
            x1 = np.random.randint(0, w - 200)
            y1 = np.random.randint(0, h - 150)
            x2 = x1 + np.random.randint(100, 300)
            y2 = y1 + np.random.randint(80, 200)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        writer.write(frame)

    writer.release()
    return str(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RF-DETR Nano vs Small",
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Video file. If not given, generates synthetic frames.")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Number of frames to benchmark (default: 200)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--fp16", action="store_true",
                        help="Try fp16 inference (if supported)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RF-DETR BENCHMARK")
    print(f"{'='*60}")

    # System info
    ram_avail = psutil.virtual_memory().available / (1024 * 1024)
    print(f"  RAM available: {ram_avail:.0f} MB")
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")

    # Source
    if args.source:
        source = args.source
    else:
        source = generate_test_frames(args.n_frames)

    # Import RF-DETR variants
    from rfdetr import RFDETRNano

    results = []

    # Benchmark Nano
    r_nano = benchmark_model(RFDETRNano, "RF-DETR-Nano", source,
                             n_frames=args.n_frames, threshold=args.threshold)
    results.append(r_nano)

    # Benchmark Small (if Nano didn't exhaust VRAM)
    try:
        from rfdetr import RFDETRSmall
        r_small = benchmark_model(RFDETRSmall, "RF-DETR-Small", source,
                                  n_frames=args.n_frames, threshold=args.threshold)
        results.append(r_small)
    except ImportError:
        print("\n  RFDETRSmall not available in this rfdetr version")
    except Exception as e:
        print(f"\n  RFDETRSmall benchmark failed: {e}")
        results.append({"model": "RF-DETR-Small", "error": str(e)})

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Nano':>15} {'Small':>15}")
    print(f"  {'-'*55}")

    metrics = [
        ("VRAM peak (MB)", "vram_peak_mb"),
        ("Inference median (ms)", ("inference_ms", "median")),
        ("FPS sustained", "fps_sustained"),
        ("Vehicles/frame", "vehicles_per_frame_mean"),
        ("GPU temp max (C)", "gpu_temp_max_c"),
        ("RSS (MB)", "rss_mb"),
    ]

    for label, key in metrics:
        vals = []
        for r in results:
            if "error" in r:
                vals.append("ERROR")
            elif isinstance(key, tuple):
                v = r.get(key[0], {}).get(key[1], "?")
                vals.append(str(v))
            else:
                vals.append(str(r.get(key, "?")))
        row = f"  {label:<25}"
        for v in vals:
            row += f" {v:>15}"
        print(row)

    # Decision
    print(f"\n  --- DECISION ---")
    nano_ok = not any("error" in r for r in results if r.get("model", "").startswith("RF-DETR-Nano"))
    if nano_ok:
        nano = [r for r in results if "Nano" in r.get("model", "")][0]
        nano_vram = nano.get("vram_peak_mb", 9999)
        nano_fps = nano.get("fps_sustained", 0)

        if nano_vram > 4000:
            print(f"  WARNING: Nano VRAM peak ({nano_vram:.0f} MB) exceeds 4 GB budget!")
        if nano_fps < 12:
            print(f"  WARNING: Nano FPS ({nano_fps:.1f}) below 12 FPS target!")

        small_results = [r for r in results if "Small" in r.get("model", "") and "error" not in r]
        if small_results:
            small = small_results[0]
            # Choose Small only if recall improvement > 2 pts AND FPS >= 12
            # Since we can't measure recall without labels, use vehicles_per_frame as proxy
            nano_vpf = nano.get("vehicles_per_frame_mean", 0)
            small_vpf = small.get("vehicles_per_frame_mean", 0)
            small_fps = small.get("fps_sustained", 0)
            small_vram = small.get("vram_peak_mb", 9999)

            if small_vpf > nano_vpf * 1.02 and small_fps >= 12 and small_vram <= 4000:
                print(f"  RECOMMENDATION: Use RF-DETR-Small "
                      f"(+{((small_vpf/nano_vpf)-1)*100:.1f}% detection, {small_fps:.1f} FPS)")
            else:
                print(f"  RECOMMENDATION: Use RF-DETR-Nano "
                      f"(within budget, {nano_fps:.1f} FPS, {nano_vram:.0f} MB VRAM)")
        else:
            print(f"  RECOMMENDATION: Use RF-DETR-Nano ({nano_fps:.1f} FPS, {nano_vram:.0f} MB VRAM)")

    # Save results
    out_path = Path(__file__).parent.parent / "eval" / "rfdetr_benchmark.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
