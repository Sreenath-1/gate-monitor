"""
tools/record_clips.py
=====================
Record N-minute clips from a camera key into data/clips/.

Usage:
    python tools/record_clips.py --camera cctv_gate --duration 5
    python tools/record_clips.py --camera phone_test --duration 2 --count 3
    python tools/record_clips.py --source rtsp://... --duration 5

Clips are saved as:
    data/clips/{camera_id}_{YYYYMMDD_HHMMSS}.mp4

The data/clips/ directory is gitignored.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from config import IST

# Minimum free disk space (bytes) before refusing to record
MIN_FREE_DISK_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB


def get_free_disk_gb(path: str) -> float:
    """Return free disk space in GB for the drive containing `path`."""
    import shutil
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def resolve_camera_url(camera_id: str, cameras_yaml: str) -> str:
    """Look up the RTSP URL for a camera_id from cameras.yaml.

    Resolves ${ENV_VAR} references against os.environ.
    Prefers hires_stream; falls back to detect_stream.
    """
    import re
    import yaml

    with open(cameras_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    cameras = cfg.get("cameras", [])
    cam = None
    for c in cameras:
        if c.get("camera_id") == camera_id:
            cam = c
            break

    if cam is None:
        available = [c.get("camera_id", "?") for c in cameras]
        raise ValueError(
            f"Camera '{camera_id}' not found in {cameras_yaml}. "
            f"Available: {available}"
        )

    # Prefer hires_stream, fall back to detect_stream
    url = cam.get("hires_stream") or cam.get("detect_stream")
    if url is None:
        raise ValueError(f"Camera '{camera_id}' has no stream URL configured.")

    # Resolve ${ENV_VAR} references
    def _env_sub(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(
                f"Environment variable ${{{var_name}}} referenced by camera "
                f"'{camera_id}' is not set. Add it to .env or export it."
            )
        return val

    url = re.sub(r"\$\{(\w+)\}", _env_sub, str(url))

    if url == "null" or url == "None" or not url:
        raise ValueError(
            f"Camera '{camera_id}' stream URL resolved to empty/null. "
            f"Set the appropriate env var in .env."
        )

    return url


def record_clip(
    source: str,
    output_path: str,
    duration_secs: float,
    camera_id: str = "unknown",
) -> dict:
    """Record a single clip from an RTSP/file source.

    Args:
        source: RTSP URL or file path.
        output_path: Where to write the .mp4 clip.
        duration_secs: How many seconds to record.
        camera_id: For logging.

    Returns:
        Dict with recording stats.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Force TCP for RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return {"success": False, "error": f"Cannot open: {source}"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        cap.release()
        return {"success": False, "error": f"Cannot create writer: {output_path}"}

    frames_written = 0
    t_start = time.time()
    errors = 0

    print(f"  ● Recording {camera_id} → {output_path}")
    print(f"    Resolution: {w}×{h} @ {fps:.1f} FPS, duration: {duration_secs}s")

    while time.time() - t_start < duration_secs:
        ret, frame = cap.read()
        if not ret:
            errors += 1
            if errors > 50:
                break
            time.sleep(0.01)
            continue
        writer.write(frame)
        frames_written += 1
        errors = 0

    elapsed = time.time() - t_start
    cap.release()
    writer.release()

    actual_fps = frames_written / elapsed if elapsed > 0 else 0

    result = {
        "success": frames_written > 0,
        "output": output_path,
        "camera_id": camera_id,
        "resolution": f"{w}x{h}",
        "nominal_fps": fps,
        "actual_fps": round(actual_fps, 1),
        "frames": frames_written,
        "duration_secs": round(elapsed, 1),
        "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 1)
        if os.path.exists(output_path) else 0,
    }

    if result["success"]:
        print(f"    ✓ {frames_written} frames, {result['file_size_mb']} MB, "
              f"{actual_fps:.1f} FPS actual")
    else:
        print(f"    ✗ Recording failed after {errors} consecutive errors")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Record clips from cameras into data/clips/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python tools/record_clips.py --camera cctv_gate --duration 5
  python tools/record_clips.py --camera phone_test --duration 2 --count 3
  python tools/record_clips.py --source "rtsp://..." --duration 5
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", type=str,
                       help="Camera ID from cameras.yaml")
    group.add_argument("--source", type=str,
                       help="Direct RTSP URL or file path")

    parser.add_argument("--duration", type=float, default=5.0,
                        help="Clip duration in minutes (default: 5)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of clips to record (default: 1)")
    parser.add_argument("--cameras-yaml", type=str,
                        default=str(Path(__file__).parent.parent / "cameras.yaml"),
                        help="Path to cameras.yaml")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "clips"),
                        help="Output directory (default: data/clips/)")

    args = parser.parse_args()

    # Check free disk space
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    free_gb = get_free_disk_gb(output_dir)
    if free_gb < 20.0:
        print(f"  ✗ REFUSING to record: only {free_gb:.1f} GB free "
              f"(minimum 20 GB required)")
        sys.exit(1)

    # Resolve source URL
    if args.camera:
        # Load .env for env var resolution
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            from config import _load_dotenv
            _load_dotenv(env_path)

        source = resolve_camera_url(args.camera, args.cameras_yaml)
        camera_id = args.camera
    else:
        source = args.source
        camera_id = "direct"

    duration_secs = args.duration * 60

    print(f"\n{'='*60}")
    print(f"  CLIP RECORDER")
    print(f"{'='*60}")
    print(f"  Camera  : {camera_id}")
    print(f"  Duration: {args.duration} min × {args.count} clip(s)")
    print(f"  Disk    : {free_gb:.1f} GB free")
    print(f"  Output  : {output_dir}")
    print()

    results = []
    for i in range(args.count):
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{ts}.mp4"
        output_path = os.path.join(output_dir, filename)

        result = record_clip(source, output_path, duration_secs, camera_id)
        results.append(result)

        # Check disk again between clips
        free_gb = get_free_disk_gb(output_dir)
        if free_gb < 20.0:
            print(f"\n  ✗ Stopping: disk below 20 GB free ({free_gb:.1f} GB)")
            break

        if i < args.count - 1:
            print(f"  ... waiting 2s before next clip ...")
            time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    ok = sum(1 for r in results if r["success"])
    total_mb = sum(r.get("file_size_mb", 0) for r in results)
    print(f"  Recorded {ok}/{len(results)} clips, {total_mb:.1f} MB total")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
