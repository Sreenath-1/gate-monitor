"""
camera_manager.py
=================
Multi-camera source management for the person search pipeline.

Provides:
  CameraConfig   — Parsed camera definition from cameras.yaml
  CameraSource   — Single camera with health monitoring, latest-frame-wins
  CameraManager  — Manages all configured cameras, health polling, diagnostics

Reuses the existing ThreadedCapture and FrameSource from stream.py for the
actual capture/reconnect logic. This module adds the multi-camera, health,
and configuration layers on top.

Design:
  - Each camera runs its own ThreadedCapture (via FrameSource) in a daemon thread
  - Latest-frame-wins: never queue stale frames; consumer grabs latest via read()
  - Health states: ONLINE / DEGRADED / OFFLINE with frame-age watchdog
  - Webcam warmup: discard first ~10 frames (auto-exposure settling)
  - All camera URLs resolve .env variable references at load time
"""

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

import cv2
import yaml
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IST, RTSP_RECONNECT_DELAY, RTSP_MAX_RECONNECT_SECS


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH STATES
# ─────────────────────────────────────────────────────────────────────────────
class CameraHealth(Enum):
    """Camera connection health state."""
    ONLINE   = "ONLINE"     # Receiving fresh frames
    DEGRADED = "DEGRADED"   # Receiving frames but with issues (low FPS, stale)
    OFFLINE  = "OFFLINE"    # Not receiving frames
    DISABLED = "DISABLED"   # Configured but not enabled


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA CONFIG (from cameras.yaml)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """Parsed camera definition from cameras.yaml."""
    camera_id: str
    name: str
    type: str                        # "webcam", "rtsp", "http"
    location_label: str
    enabled: bool
    clock_offset_s: float
    detect_stream: Any               # int for webcam index, str for URL
    hires_stream: Optional[str]      # Optional hi-res URL for on-hit grabs


def _resolve_env_vars(value):
    """Resolve ${VAR_NAME} references in a string using os.environ.

    Returns the resolved string, or the original value if no vars found
    or if the referenced var is not set.
    """
    if not isinstance(value, str):
        return value
    if "${" not in value:
        return value

    import re
    def _replace(match):
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            return match.group(0)  # Leave unresolved
        return env_val

    return re.sub(r'\$\{([^}]+)\}', _replace, value)


def load_camera_configs(yaml_path: str = None) -> List[CameraConfig]:
    """Load camera configurations from cameras.yaml.

    Args:
        yaml_path: Path to cameras.yaml. Defaults to project_root/cameras.yaml.

    Returns:
        List of CameraConfig objects.

    Raises:
        FileNotFoundError: If cameras.yaml doesn't exist.
        ValueError: If required fields are missing.
    """
    if yaml_path is None:
        yaml_path = str(Path(__file__).parent / "cameras.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Camera config not found: {yaml_path}\n"
            f"Copy cameras.yaml.example to cameras.yaml and edit for your setup."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "cameras" not in data:
        raise ValueError(f"cameras.yaml must contain a 'cameras' list")

    configs = []
    seen_ids = set()

    for i, cam in enumerate(data["cameras"]):
        # Validate required fields
        cam_id = cam.get("camera_id")
        if not cam_id:
            raise ValueError(f"Camera #{i+1}: missing 'camera_id'")
        if cam_id in seen_ids:
            raise ValueError(f"Duplicate camera_id: '{cam_id}'")
        seen_ids.add(cam_id)

        cam_type = cam.get("type", "").lower()
        if cam_type not in ("webcam", "rtsp", "http", "file"):
            raise ValueError(
                f"Camera '{cam_id}': invalid type '{cam_type}'. "
                f"Must be: webcam, rtsp, http, file"
            )

        # Resolve detect_stream
        detect_stream = cam.get("detect_stream")
        if detect_stream is None:
            raise ValueError(f"Camera '{cam_id}': missing 'detect_stream'")

        if cam_type == "webcam":
            # Webcam: detect_stream should be an integer index
            detect_stream = int(detect_stream)
        else:
            # RTSP/HTTP: resolve env vars in URL
            detect_stream = _resolve_env_vars(str(detect_stream))

        # Resolve hires_stream (optional)
        hires_stream = cam.get("hires_stream")
        if hires_stream is not None:
            hires_stream = _resolve_env_vars(str(hires_stream))
            if "${" in hires_stream:
                # Env var not resolved — treat as not configured
                hires_stream = None

        configs.append(CameraConfig(
            camera_id=cam_id,
            name=cam.get("name", cam_id),
            type=cam_type,
            location_label=cam.get("location_label", ""),
            enabled=cam.get("enabled", True),
            clock_offset_s=float(cam.get("clock_offset_s", 0)),
            detect_stream=detect_stream,
            hires_stream=hires_stream,
        ))

    return configs


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA SOURCE (single camera with health monitoring)
# ─────────────────────────────────────────────────────────────────────────────

# Frame-age thresholds for health classification
_STALE_FRAME_SECS = 5.0       # If last frame is older than this → DEGRADED
_OFFLINE_FRAME_SECS = 15.0    # If last frame is older than this → OFFLINE
_WEBCAM_WARMUP_FRAMES = 10    # Discard first N frames for auto-exposure


class CameraSource:
    """A single camera source with health monitoring and latest-frame-wins.

    Wraps the existing ThreadedCapture (for streams) or cv2.VideoCapture
    (for webcam) with added health tracking and per-camera metadata.

    Usage:
        cam = CameraSource(config)
        cam.start()
        ...
        ret, frame, ts = cam.read()
        health = cam.health
        ...
        cam.stop()
    """

    def __init__(self, config: CameraConfig, resize_height: int = 720):
        self.config = config
        self.camera_id = config.camera_id
        self.resize_height = resize_height

        # State
        self._health = CameraHealth.DISABLED if not config.enabled else CameraHealth.OFFLINE
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ret = False
        self._latest_ts = None          # Wall-clock timestamp of last frame
        self._frame_count = 0
        self._start_time = None
        self._stopped = False
        self._reader_thread = None
        self._error_message = None

        # Resolution info (populated after start)
        self.raw_width = 0
        self.raw_height = 0
        self.raw_fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self._do_resize = False

    @property
    def health(self) -> CameraHealth:
        """Current health state, updated based on frame freshness."""
        if self._health == CameraHealth.DISABLED:
            return CameraHealth.DISABLED
        if self._latest_ts is None:
            return CameraHealth.OFFLINE

        age = time.time() - self._latest_ts
        if age > _OFFLINE_FRAME_SECS:
            return CameraHealth.OFFLINE
        elif age > _STALE_FRAME_SECS:
            return CameraHealth.DEGRADED
        return CameraHealth.ONLINE

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def fps_measured(self) -> float:
        """Measured FPS since start."""
        if self._start_time is None or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def start(self) -> bool:
        """Open the camera and start reading frames.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if not self.config.enabled:
            self._health = CameraHealth.DISABLED
            return False

        self._stopped = False
        self._error_message = None

        try:
            success = self._open_capture()
            if not success:
                return False

            self._start_time = time.time()
            self._health = CameraHealth.ONLINE

            # Start background reader thread
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name=f"cam-{self.camera_id}",
                daemon=True,
            )
            self._reader_thread.start()
            return True

        except Exception as e:
            self._error_message = str(e)
            self._health = CameraHealth.OFFLINE
            return False

    def _open_capture(self) -> bool:
        """Open the underlying capture device/stream."""
        cfg = self.config
        source = cfg.detect_stream

        # Check for unresolved env vars
        if isinstance(source, str) and "${" in source:
            self._error_message = (
                f"Unresolved env variable in URL for camera '{cfg.camera_id}'. "
                f"Check .env file. URL: {source}"
            )
            return False

        if cfg.type == "webcam":
            # Webcam: use DirectShow on Windows for reliability
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default backend
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self._error_message = (
                    f"Cannot open webcam index {source}. "
                    f"Device may be in use by another app or index doesn't exist."
                )
                return False

            # Warmup: discard first frames (auto-exposure settling)
            for _ in range(_WEBCAM_WARMUP_FRAMES):
                cap.read()

        elif cfg.type in ("rtsp", "http"):
            # Network stream: force TCP for RTSP
            if cfg.type == "rtsp":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self._error_message = f"Cannot open stream: {source}"
                return False

        elif cfg.type == "file":
            # Local video file (used for testing / replay)
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self._error_message = f"Cannot open video file: {source}"
                return False

        else:
            self._error_message = f"Unknown camera type: {cfg.type}"
            return False

        # Read native properties
        self.raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Sanitize FPS for streams
        if cfg.type != "webcam" and (self.raw_fps <= 0 or self.raw_fps > 120):
            self.raw_fps = 25.0

        # Resize configuration
        rh = self.resize_height
        self._do_resize = rh > 0 and self.raw_height > rh
        if self._do_resize:
            scale = rh / self.raw_height
            self.frame_width = int(self.raw_width * scale)
            self.frame_height = rh
        else:
            self.frame_width = self.raw_width
            self.frame_height = self.raw_height

        # Read first frame to confirm
        ret, frame = cap.read()
        if not ret:
            self._error_message = f"Opened but first read failed: {source}"
            cap.release()
            return False

        if self._do_resize:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        with self._lock:
            self._cap = cap
            self._latest_ret = True
            self._latest_frame = frame
            self._latest_ts = time.time()
            self._frame_count = 1

        return True

    def _reader_loop(self):
        """Background thread: continuously read frames, keep latest only.

        For streams: handles reconnection via exponential backoff.
        For webcam: simpler read loop (no reconnect, just retry).
        """
        consecutive_failures = 0
        reconnect_start = None
        reconnect_attempts = 0

        while not self._stopped:
            try:
                ret, frame = self._cap.read()
            except cv2.error:
                if self._stopped:
                    return
                ret, frame = False, None
            except Exception:
                if self._stopped:
                    return
                ret, frame = False, None

            if ret and frame is not None:
                # Apply resize
                if self._do_resize:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                # Apply clock offset to timestamp
                ts = time.time() + self.config.clock_offset_s

                with self._lock:
                    self._latest_ret = True
                    self._latest_frame = frame
                    self._latest_ts = ts
                    self._frame_count += 1

                consecutive_failures = 0
                reconnect_start = None
            else:
                if self._stopped:
                    return
                consecutive_failures += 1

                if consecutive_failures >= 10 and self.config.type != "webcam":
                    # Stream dropped — attempt reconnect
                    if reconnect_start is None:
                        reconnect_start = time.time()

                    elapsed = time.time() - reconnect_start
                    if elapsed > RTSP_MAX_RECONNECT_SECS:
                        self._error_message = (
                            f"Reconnect failed after {elapsed:.0f}s"
                        )
                        with self._lock:
                            self._latest_ret = False
                        return

                    delay = min(RTSP_RECONNECT_DELAY * (2 ** min(reconnect_attempts, 3)), 16)
                    reconnect_attempts += 1
                    time.sleep(delay)

                    # Try to reopen
                    try:
                        self._cap.release()
                    except Exception:
                        pass

                    source = self.config.detect_stream
                    if self.config.type == "rtsp":
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    new_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if new_cap.isOpened():
                        ret2, frame2 = new_cap.read()
                        if ret2:
                            with self._lock:
                                self._cap = new_cap
                                self._latest_ret = True
                                self._latest_frame = frame2
                                self._latest_ts = time.time()
                            consecutive_failures = 0
                            reconnect_start = None
                            reconnect_attempts = 0
                        else:
                            new_cap.release()
                    # If reopen failed, loop will retry
                elif consecutive_failures >= 30 and self.config.type == "webcam":
                    # Webcam: too many failures = device lost
                    self._error_message = "Webcam read failed repeatedly"
                    with self._lock:
                        self._latest_ret = False
                    return
                else:
                    time.sleep(0.01)

    def read(self):
        """Return the latest frame with timestamp (thread-safe).

        Returns:
            (bool, frame_or_None, timestamp_float_or_None)
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_ret, self._latest_frame.copy(), self._latest_ts
            return False, None, None

    def stop(self):
        """Stop reading and release the capture."""
        self._stopped = True
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._health = CameraHealth.OFFLINE

    def info_dict(self) -> dict:
        """Return camera info for diagnostics/display."""
        return {
            "camera_id": self.camera_id,
            "name": self.config.name,
            "type": self.config.type,
            "location": self.config.location_label,
            "enabled": self.config.enabled,
            "health": self.health.value,
            "raw_resolution": f"{self.raw_width}x{self.raw_height}",
            "process_resolution": f"{self.frame_width}x{self.frame_height}",
            "raw_fps": self.raw_fps,
            "measured_fps": round(self.fps_measured, 1),
            "frames_read": self._frame_count,
            "error": self._error_message,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA MANAGER (manages all configured cameras)
# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    """Manages multiple CameraSource instances from cameras.yaml.

    Usage:
        mgr = CameraManager("cameras.yaml")
        mgr.start_all()
        ...
        for cam_id, cam in mgr.cameras.items():
            ret, frame, ts = cam.read()
        ...
        mgr.stop_all()
    """

    def __init__(self, yaml_path: str = None, resize_height: int = 720):
        """Load camera configs and create CameraSource objects.

        Args:
            yaml_path: Path to cameras.yaml.
            resize_height: Target height for detection frames.
        """
        self.configs = load_camera_configs(yaml_path)
        self.cameras: Dict[str, CameraSource] = {}
        self.resize_height = resize_height

        for cfg in self.configs:
            self.cameras[cfg.camera_id] = CameraSource(cfg, resize_height)

    @property
    def enabled_cameras(self) -> Dict[str, CameraSource]:
        """Return only cameras that are enabled in config."""
        return {
            cid: cam for cid, cam in self.cameras.items()
            if cam.config.enabled
        }

    def start_all(self) -> Dict[str, bool]:
        """Start all enabled cameras.

        Returns:
            Dict mapping camera_id → success (True/False).
        """
        results = {}
        for cam_id, cam in self.cameras.items():
            if not cam.config.enabled:
                results[cam_id] = False
                continue
            success = cam.start()
            results[cam_id] = success
        return results

    def stop_all(self):
        """Stop all cameras gracefully."""
        for cam in self.cameras.values():
            cam.stop()

    def health_report(self) -> List[dict]:
        """Get health info for all cameras (for diagnostics/UI)."""
        report = []
        for cam in self.cameras.values():
            info = cam.info_dict()
            report.append(info)
        return report

    def print_health_report(self):
        """Print a formatted health report to stdout."""
        report = self.health_report()

        print(f"\n{'='*70}")
        print(f"  CAMERA HEALTH REPORT")
        print(f"{'='*70}")

        for info in report:
            health = info["health"]
            # Color-code health
            if health == "ONLINE":
                icon = "✓"
            elif health == "DEGRADED":
                icon = "⚠"
            elif health == "OFFLINE":
                icon = "✗"
            else:
                icon = "○"

            print(f"\n  {icon}  {info['name']} ({info['camera_id']})")
            print(f"     Type     : {info['type']}")
            print(f"     Location : {info['location']}")
            print(f"     Health   : {health}")
            print(f"     Enabled  : {info['enabled']}")

            if health == "ONLINE" or health == "DEGRADED":
                print(f"     Native   : {info['raw_resolution']} @ {info['raw_fps']:.1f} fps")
                print(f"     Process  : {info['process_resolution']}")
                print(f"     Measured : {info['measured_fps']:.1f} fps ({info['frames_read']} frames)")

            if info.get("error"):
                print(f"     Error    : {info['error']}")

        online = sum(1 for i in report if i["health"] == "ONLINE")
        total = sum(1 for i in report if i["enabled"])
        print(f"\n  Summary: {online}/{total} cameras online")
        print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCES-CHECK MODE (diagnostic — run cameras for N seconds and report)
# ─────────────────────────────────────────────────────────────────────────────
def run_sources_check(yaml_path: str = None, duration: int = 30,
                      resize_height: int = 720):
    """Run all cameras for `duration` seconds and report health.

    This is the implementation of `--mode sources-check`.

    Args:
        yaml_path: Path to cameras.yaml.
        duration: How long to run in seconds.
        resize_height: Frame resize height.
    """
    print(f"\n{'='*70}")
    print(f"  CAMERA SOURCES CHECK (running for {duration}s)")
    print(f"{'='*70}\n")

    try:
        mgr = CameraManager(yaml_path, resize_height)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    # Start all cameras
    results = mgr.start_all()
    for cam_id, success in results.items():
        cam = mgr.cameras[cam_id]
        if not cam.config.enabled:
            print(f"  ○ {cam.config.name} ({cam_id}) — DISABLED")
        elif success:
            print(f"  ✓ {cam.config.name} ({cam_id}) — opened "
                  f"{cam.raw_width}x{cam.raw_height} @ {cam.raw_fps:.0f}fps")
        else:
            print(f"  ✗ {cam.config.name} ({cam_id}) — FAILED: {cam.error_message}")

    # Run for duration, showing intermediate health
    t_start = time.time()
    show_window = True

    try:
        while time.time() - t_start < duration:
            elapsed = time.time() - t_start

            # Show combined preview window (all cameras tiled)
            frames_to_show = []
            for cam_id, cam in mgr.enabled_cameras.items():
                ret, frame, ts = cam.read()
                if ret and frame is not None:
                    # Add label overlay
                    label = f"{cam.config.name} [{cam.health.value}]"
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if cam.health == CameraHealth.ONLINE else (0, 0, 255),
                                2)
                    fps_label = f"FPS: {cam.fps_measured:.1f}"
                    cv2.putText(frame, fps_label, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    frames_to_show.append(frame)

            if frames_to_show and show_window:
                # Tile frames horizontally (resize each to same height)
                target_h = 360
                resized = []
                for f in frames_to_show:
                    scale = target_h / f.shape[0]
                    w = int(f.shape[1] * scale)
                    resized.append(cv2.resize(f, (w, target_h)))
                tiled = np.hstack(resized)
                cv2.imshow("Camera Sources Check [ESC=stop]", tiled)

            key = cv2.waitKey(100)  # 10 FPS preview refresh
            if key == 27:  # ESC
                print("\n  Stopped by user.")
                break

            # Print periodic health
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                # Only print once per 10s mark
                if abs(elapsed - int(elapsed)) < 0.15:
                    online = sum(
                        1 for c in mgr.enabled_cameras.values()
                        if c.health == CameraHealth.ONLINE
                    )
                    total = len(mgr.enabled_cameras)
                    print(f"  [{elapsed:.0f}s] {online}/{total} cameras online")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        cv2.destroyAllWindows()
        mgr.stop_all()

    # Final report
    mgr.print_health_report()


# ─────────────────────────────────────────────────────────────────────────────
#  WEBCAM INDEX PROBE (discover available webcams)
# ─────────────────────────────────────────────────────────────────────────────
def probe_webcam_indices(max_index: int = 5) -> List[dict]:
    """Probe webcam device indices 0..max_index and report which open.

    Returns:
        List of dicts with 'index', 'resolution', 'fps', 'backend'.
    """
    found = []
    for idx in range(max_index):
        # Try DirectShow first (Windows)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            backend = "DEFAULT"

        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            cap.release()
            if ret:
                found.append({
                    "index": idx,
                    "resolution": f"{w}x{h}",
                    "fps": fps,
                    "backend": backend,
                })
        else:
            cap.release()

    return found
