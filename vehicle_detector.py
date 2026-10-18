"""
vehicle_detector.py
===================
Vehicle detection using RF-DETR Nano with supervision.Detections output.

Responsibilities:
  - Load RF-DETR Nano (COCO pretrained)
  - Filter to vehicle classes only (car, motorcycle, bus, truck)
  - Expose GPU adapter check + fail-fast on --require-gpu
  - Track VRAM usage and GPU temperature for adaptive load-shedding
  - Provide consistent supervision.Detections interface for downstream

Architecture:
  Frame → RF-DETR Nano → filter vehicle classes → sv.Detections

Usage:
    detector = VehicleDetector(conf_threshold=0.35, require_gpu=True)
    detections = detector.detect(frame)  # sv.Detections
    detector.close()
"""

import logging
import os
import subprocess
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("vehicle_detector")

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Full COCO class list for reference
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
}


class VehicleDetector:
    """RF-DETR Nano vehicle detector with CUDA enforcement and thermal awareness.

    Args:
        conf_threshold: Minimum detection confidence (default: 0.35).
        require_gpu: If True, fail fast if CUDA is not available.
        gpu_temp_limit: GPU temperature threshold for load-shedding (default: 80°C).
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        require_gpu: bool = False,
        gpu_temp_limit: int = 80,
    ):
        self.conf_threshold = conf_threshold
        self.gpu_temp_limit = gpu_temp_limit
        self._throttled = False
        self._model = None

        # GPU check
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "--require-gpu: CUDA is not available. "
                "Check driver, torch installation, and GPU visibility."
            )

        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA device: %s", self._device_name)
        else:
            self._device_name = "CPU"
            logger.warning("No CUDA — running on CPU (will be slow)")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load RF-DETR Nano model."""
        t0 = time.time()
        try:
            from rfdetr import RFDETRNano
            self._model = RFDETRNano()
            logger.info("RF-DETR Nano loaded in %.1fs", time.time() - t0)
        except Exception as e:
            logger.error("Failed to load RF-DETR Nano: %s", e)
            raise

        # Log VRAM after load
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info("VRAM after model load: %.0f MB", vram_mb)

    def detect(self, frame: np.ndarray) -> "supervision.Detections":
        """Run vehicle detection on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            supervision.Detections filtered to vehicle classes only.
            Empty Detections if no vehicles found or if throttled.
        """
        import supervision as sv

        if self._model is None:
            return sv.Detections.empty()

        # Thermal check (sampled, not every frame)
        if self._throttled:
            return sv.Detections.empty()

        try:
            detections = self._model.predict(frame, threshold=self.conf_threshold)
        except Exception as e:
            logger.error("Detection failed: %s", e)
            return sv.Detections.empty()

        # Filter to vehicle classes only
        if detections.class_id is None or len(detections.class_id) == 0:
            return sv.Detections.empty()

        vehicle_mask = np.array([
            int(cid) in VEHICLE_CLASS_IDS
            for cid in detections.class_id
        ])

        if not vehicle_mask.any():
            return sv.Detections.empty()

        return detections[vehicle_mask]

    def check_thermal(self) -> dict:
        """Check GPU temperature and throttle status.

        Call this periodically (e.g. every 20 frames), not every frame.

        Returns:
            Dict with gpu_temp_c, throttled, power_w keys.
        """
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,clocks_throttle_reasons.active,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=3
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            temp = int(parts[0])
            throttle = parts[1] if len(parts) > 1 else "0x0"

            status = {
                "gpu_temp_c": temp,
                "throttle_raw": throttle,
                "throttled": False,
                "power_w": None,
            }

            try:
                status["power_w"] = float(parts[2]) if len(parts) > 2 else None
            except (ValueError, TypeError):
                pass

            # Adaptive load-shedding
            if temp > self.gpu_temp_limit:
                if not self._throttled:
                    logger.warning(
                        "GPU temp %d°C exceeds limit %d°C — enabling load-shedding",
                        temp, self.gpu_temp_limit
                    )
                self._throttled = True
                status["throttled"] = True
            elif self._throttled and temp < self.gpu_temp_limit - 5:
                # Hysteresis: restore when 5°C below limit
                logger.info("GPU temp %d°C — restoring full processing", temp)
                self._throttled = False
                status["throttled"] = False

            return status

        except Exception as e:
            logger.debug("nvidia-smi check failed: %s", e)
            return {"gpu_temp_c": None, "throttled": False}

    @property
    def is_throttled(self) -> bool:
        """True if adaptive load-shedding is active."""
        return self._throttled

    @property
    def vram_mb(self) -> float:
        """Current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    @property
    def vram_peak_mb(self) -> float:
        """Peak VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def get_class_name(self, class_id: int) -> str:
        """Map COCO class ID to vehicle type name."""
        return VEHICLE_CLASS_NAMES.get(int(class_id), f"class_{class_id}")

    def close(self):
        """Release model and free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VehicleDetector closed")

    def __del__(self):
        self.close()
