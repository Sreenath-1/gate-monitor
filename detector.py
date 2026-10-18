"""
detector.py
============
YOLOv11 person detection wrapper.

Wraps Ultralytics YOLO model for person-only detection. Handles model loading,
inference, and result extraction. The tracking integration is kept here (not in
a separate tracker module) because Ultralytics couples detection and tracking
in a single `model.track()` call — splitting them would require re-implementing
the tracker from scratch for no benefit.

Performance note (imgsz):
    YOLO's `imgsz` controls the internal inference resolution, separate from the
    frame's pixel resolution. Smaller imgsz = faster inference at the cost of
    small-object detection. For person-sized targets at CCTV distances, imgsz=480
    provides ~3-4x speedup over the default 640 with acceptable accuracy.
    The frame is already resized to 720p before reaching YOLO, so imgsz=480 means
    YOLO internally downscales from 720p → 480p for inference, then maps boxes
    back to 720p coordinates.
"""

import time
from dataclasses import dataclass, field
from typing import List

from ultralytics import YOLO

from config import PERSON_CLASS, CONF_THRESHOLD, IOU_THRESHOLD, DEFAULT_TRACKER, DEFAULT_IMGSZ


@dataclass
class Detection:
    """A single detected + tracked person in one frame."""
    track_id: int           # Persistent tracking ID across frames
    x1: int                 # Bounding box top-left X
    y1: int                 # Bounding box top-left Y
    x2: int                 # Bounding box bottom-right X
    y2: int                 # Bounding box bottom-right Y
    cx: int                 # Center X (centroid)
    cy: int                 # Center Y (centroid)
    confidence: float       # Detection confidence score


@dataclass
class FrameResult:
    """Detection + tracking results for one frame."""
    detections: List[Detection] = field(default_factory=list)
    active_track_ids: set = field(default_factory=set)
    inference_ms: float = 0.0     # Time spent on YOLO inference


class PersonDetector:
    """YOLOv11 person detector with integrated tracking.

    Usage:
        detector = PersonDetector("yolo11n.pt", tracker="botsort.yaml", imgsz=480)
        result = detector.detect(frame)
        for det in result.detections:
            print(f"Person #{det.track_id} at ({det.cx}, {det.cy}) conf={det.confidence:.2f}")
    """

    def __init__(self, weights: str, tracker: str = DEFAULT_TRACKER,
                 conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD,
                 imgsz: int = DEFAULT_IMGSZ):
        """
        Args:
            weights: Path to YOLO weights file (e.g. "yolo11n.pt").
            tracker: Tracker config name ("botsort.yaml" or "bytetrack.yaml").
            conf: Detection confidence threshold.
            iou: NMS IoU threshold.
            imgsz: YOLO internal inference resolution. Smaller = faster.
                   480 is ~3x faster than 640 for person detection at CCTV range.
        """
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = YOLO(weights)
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda')
        self._weights_name = weights

    @property
    def weights_name(self):
        return self._weights_name

    def detect(self, frame) -> FrameResult:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR numpy array (OpenCV frame).

        Returns:
            FrameResult with list of Detection objects and timing info.
        """
        t0 = time.time()
        results = self.model.track(
            frame,
            persist=True,           # Maintain tracking state across calls
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,       # Internal inference resolution (speed vs accuracy)
            classes=[PERSON_CLASS],  # Person class only
            tracker=self.tracker,
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        r = results[0]
        frame_result = FrameResult(inference_ms=inference_ms)

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids   = r.boxes.id.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                det = Detection(
                    track_id=int(tid),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                    confidence=float(conf),
                )
                frame_result.detections.append(det)
                frame_result.active_track_ids.add(int(tid))

        return frame_result

    def detect_no_track(self, frame) -> FrameResult:
        """Run detection only (no tracking) — useful for benchmarking/testing.

        Args:
            frame: BGR numpy array.

        Returns:
            FrameResult with detections (track_id will be -1 for all).
        """
        t0 = time.time()
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[PERSON_CLASS],
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        r = results[0]
        frame_result = FrameResult(inference_ms=inference_ms)

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                det = Detection(
                    track_id=-1,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                    confidence=float(conf),
                )
                frame_result.detections.append(det)

        return frame_result
