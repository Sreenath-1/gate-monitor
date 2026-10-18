"""
vehicle_pipeline.py
===================
Main vehicle ANPR pipeline orchestrator (--mode vehicles).

Pipeline stages:
  1. Frame acquisition (OpenCV VideoCapture, RTSP or file)
  2. Frame skip (process every Nth frame for target FPS)
  3. Vehicle detection (RF-DETR Nano → supervision.Detections)
  4. Vehicle tracking (ByteTrack → persistent track IDs)
  5. Plate detection + OCR (per-vehicle crop)
  6. Temporal fusion (majority-vote plate text per track)
  7. Event emission (track loss → TrackEvent → SQLite + SSE)
  8. Adaptive load-shedding (GPU temp > 80°C → skip frames)

Usage:
    .venv-gpu\\Scripts\\python.exe main.py --mode vehicles --source rtsp
    .venv-gpu\\Scripts\\python.exe main.py --mode vehicles --source data/clips/test.mp4
"""

import io
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import psutil

logger = logging.getLogger("vehicle_pipeline")


class VehiclePipeline:
    """Main vehicle ANPR pipeline.

    Args:
        source: Video source (RTSP URL, file path, or camera index).
        frame_skip: Process every Nth frame (e.g. 2 = ~12 FPS from 25 FPS).
        conf_threshold: Vehicle detection confidence threshold.
        require_gpu: Fail fast if CUDA unavailable.
        gpu_temp_limit: Temperature for adaptive load-shedding (°C).
        min_plate_px: Minimum plate width for OCR.
    """

    def __init__(
        self,
        source: str,
        frame_skip: int = 2,
        conf_threshold: float = 0.35,
        require_gpu: bool = False,
        gpu_temp_limit: int = 80,
        min_plate_px: int = 60,
    ):
        self.source = source
        self.frame_skip = frame_skip
        self.conf_threshold = conf_threshold
        self.gpu_temp_limit = gpu_temp_limit
        self._running = False
        self._frame_count = 0
        self._process_count = 0
        self._event_callbacks = []

        # Lazy-loaded components
        self._detector = None
        self._tracker = None
        self._plate_reader = None
        self._cap = None

        # Stats
        self._fps_monitor = None
        self._start_time = None

        # Load components
        logger.info("Initializing vehicle pipeline...")
        logger.info("  Source: %s", source)
        logger.info("  Frame skip: %d", frame_skip)
        logger.info("  Confidence threshold: %.2f", conf_threshold)
        logger.info("  GPU temp limit: %d°C", gpu_temp_limit)
        logger.info("  Min plate width: %dpx", min_plate_px)

        self._init_detector(conf_threshold, require_gpu, gpu_temp_limit)
        self._init_tracker()
        self._init_plate_reader(min_plate_px)

    def _init_detector(self, conf_threshold, require_gpu, gpu_temp_limit):
        from vehicle_detector import VehicleDetector
        self._detector = VehicleDetector(
            conf_threshold=conf_threshold,
            require_gpu=require_gpu,
            gpu_temp_limit=gpu_temp_limit,
        )
        logger.info("  Vehicle detector ready (VRAM: %.0f MB)", self._detector.vram_mb)

    def _init_tracker(self):
        from vehicle_tracker import VehicleTracker
        self._tracker = VehicleTracker(fps=25)
        logger.info("  Vehicle tracker ready")

    def _init_plate_reader(self, min_plate_px):
        from plate_reader import PlateReader
        self._plate_reader = PlateReader(min_plate_px=min_plate_px)
        logger.info("  Plate reader ready")

    def on_event(self, callback):
        """Register a callback for track events.

        Callback signature: callback(event: TrackEvent)
        """
        self._event_callbacks.append(callback)

    def run(self):
        """Run the pipeline until stopped or source ends.

        Call stop() from another thread or signal handler to halt.
        """
        self._running = True
        self._start_time = time.time()

        # Open video source
        self._cap = self._open_source()
        if self._cap is None:
            logger.error("Cannot open source: %s", self.source)
            return

        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Source opened: %dx%d @ %.0f FPS", src_w, src_h, src_fps)
        logger.info("Effective processing rate: ~%.0f FPS (skip=%d)",
                     src_fps / self.frame_skip, self.frame_skip)

        # Check battery
        self._check_battery()

        thermal_check_interval = 20  # Check GPU temp every N processed frames

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    if self._is_file_source():
                        logger.info("End of file reached")
                        break
                    else:
                        logger.warning("Frame read failed — retrying in 1s")
                        time.sleep(1)
                        self._cap = self._open_source()
                        if self._cap is None:
                            break
                        continue

                self._frame_count += 1

                # Frame skip
                if self._frame_count % self.frame_skip != 0:
                    continue

                self._process_count += 1

                # Thermal check
                if self._process_count % thermal_check_interval == 0:
                    thermal = self._detector.check_thermal()
                    if thermal.get("throttled"):
                        logger.warning(
                            "Load-shedding active: GPU %d°C, skipping frame",
                            thermal.get("gpu_temp_c", 0)
                        )
                        continue

                # === PIPELINE STAGES ===

                # Stage 1: Vehicle detection
                t0 = time.perf_counter()
                detections = self._detector.detect(frame)
                t_detect = (time.perf_counter() - t0) * 1000

                n_vehicles = len(detections) if detections is not None and hasattr(detections, 'xyxy') else 0

                # Stage 2: Plate detection + OCR (per vehicle crop)
                plate_reads_by_det = {}
                vehicle_types = {}
                t_plate = 0

                if n_vehicles > 0 and detections.xyxy is not None:
                    t1 = time.perf_counter()
                    for i, box in enumerate(detections.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:
                            reads = self._plate_reader.read_plates(crop)
                            if reads:
                                plate_reads_by_det[i] = reads

                        # Vehicle type from detector
                        if detections.class_id is not None and i < len(detections.class_id):
                            vehicle_types[i] = self._detector.get_class_name(
                                detections.class_id[i]
                            )

                    t_plate = (time.perf_counter() - t1) * 1000

                # Stage 3: Tracking + fusion
                t2 = time.perf_counter()
                active_tracks, emitted_events = self._tracker.update(
                    detections, frame,
                    plate_reads_by_det=plate_reads_by_det,
                    vehicle_types=vehicle_types,
                )
                t_track = (time.perf_counter() - t2) * 1000

                # Stage 4: Emit events
                for event in emitted_events:
                    self._handle_event(event)

                # Logging (every 50 processed frames)
                if self._process_count % 50 == 0:
                    elapsed = time.time() - self._start_time
                    actual_fps = self._process_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Frame %d (processed %d): %d vehicles, "
                        "%d active tracks, %.1f FPS | "
                        "detect=%.0fms plate=%.0fms track=%.0fms",
                        self._frame_count, self._process_count,
                        n_vehicles, len(active_tracks), actual_fps,
                        t_detect, t_plate, t_track,
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self._cleanup()

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        """Open video source with RTSP TCP transport."""
        if self.source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        elif self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            return None

        # Set buffer size for RTSP (reduce latency)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _is_file_source(self) -> bool:
        """Check if source is a file (not RTSP or webcam)."""
        return not self.source.startswith("rtsp://") and not self.source.isdigit()

    def _check_battery(self):
        """Log battery warning if on battery power."""
        try:
            bat = psutil.sensors_battery()
            if bat and not bat.power_plugged:
                logger.warning(
                    "⚠ RUNNING ON BATTERY (%d%%) — performance will be degraded. "
                    "Plug in for full GPU power.",
                    bat.percent
                )
        except Exception:
            pass

    def _handle_event(self, event):
        """Process a completed track event."""
        logger.info(
            "EMIT track=%d plate=%s conf=%.2f valid=%s type=%s "
            "duration=%.1fs reads=%d",
            event.track_id, event.plate_text, event.plate_confidence,
            event.plate_valid, event.vehicle_type,
            event.duration_s, event.n_plate_reads,
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event callback failed: %s", e)

    def _cleanup(self):
        """Release all resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
        if self._detector is not None:
            self._detector.close()
        if self._plate_reader is not None:
            self._plate_reader.close()
        logger.info(
            "Pipeline stopped. Processed %d/%d frames",
            self._process_count, self._frame_count
        )

    def stop(self):
        """Signal the pipeline to stop."""
        self._running = False

    @property
    def stats(self) -> dict:
        """Current pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "frames_read": self._frame_count,
            "frames_processed": self._process_count,
            "elapsed_s": round(elapsed, 1),
            "fps": round(self._process_count / elapsed, 1) if elapsed > 0 else 0,
            "active_tracks": self._tracker.n_active if self._tracker else 0,
            "total_tracks": self._tracker.n_total if self._tracker else 0,
            "vram_mb": round(self._detector.vram_mb, 0) if self._detector else 0,
            "vram_peak_mb": round(self._detector.vram_peak_mb, 0) if self._detector else 0,
            "rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 0),
        }
