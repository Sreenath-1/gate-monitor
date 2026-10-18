"""
ingest_pipeline.py
==================
Always-on face observation pipeline for the person search system.

Per-camera worker threads feeding a shared inference pipeline:
  Frame → YOLO person detect → face detect (SCRFD on ROI) → quality score →
  per-track top-K best crops → quality-weighted embedding → ObservationStore

Features:
  - Adaptive detection rate: 3-6 FPS when motion, drops to 1 when idle
  - Per-track top-K crop selection (K=5) by quality score
  - Track-level quality-weighted mean embedding
  - Logs "person present, face unusable" observations (nothing silently missed)
  - IR/grayscale detection for night-mode cameras
  - Motion gate (frame differencing) to skip idle frames

CPU budget: targets ≥15 FPS aggregate across all cameras.
"""

import os
import sys
import time
import uuid
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IST
from camera_manager import CameraSource, CameraManager, CameraHealth
from face_embedder import FaceEmbedder, FaceDetection
from observation_store import ObservationStore, Observation
from quality import assess_quality, QualityResult


# ─────────────────────────────────────────────────────────────────────────────
#  PER-TRACK BEST FRAME SELECTION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackCrop:
    """A candidate crop for a tracked person."""
    frame_crop: np.ndarray       # BGR person ROI
    face_detection: FaceDetection
    quality_score: float
    embedding: Optional[np.ndarray]
    periocular_embedding: Optional[np.ndarray]
    timestamp: float
    bbox: Tuple[int, int, int, int]
    det_score: float
    face_size_px: int
    yaw: float
    pitch: float
    blur_score: float
    brightness: float
    mask_state: str
    is_ir: bool


class TrackState:
    """Per-track state: rolling top-K crops for best-frame selection."""

    def __init__(self, track_id: int, camera_id: str, top_k: int = 5):
        self.track_id = track_id
        self.camera_id = camera_id
        self.top_k = top_k
        self.crops: List[TrackCrop] = []
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.total_detections = 0
        self.face_detections = 0
        self.unusable_count = 0
        self._last_emit_time = 0.0
        self._emitted = False

    def add_crop(self, crop: TrackCrop):
        """Add a crop and keep only top-K by quality score."""
        self.crops.append(crop)
        self.crops.sort(key=lambda c: c.quality_score, reverse=True)
        if len(self.crops) > self.top_k:
            self.crops = self.crops[:self.top_k]
        self.last_seen = time.time()
        self.total_detections += 1
        if crop.embedding is not None:
            self.face_detections += 1

    def get_weighted_embedding(self) -> Optional[np.ndarray]:
        """Quality-weighted mean of top-K embeddings."""
        embs = [(c.embedding, c.quality_score) for c in self.crops
                if c.embedding is not None]
        if not embs:
            return None

        weights = np.array([q for _, q in embs], dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        weighted = np.zeros(embs[0][0].shape, dtype=np.float32)
        for (emb, _), w in zip(embs, weights):
            weighted += w * emb
        norm = np.linalg.norm(weighted)
        if norm > 0:
            weighted /= norm
        return weighted

    def get_best_periocular(self) -> Optional[np.ndarray]:
        """Best periocular embedding from top-K."""
        for c in self.crops:
            if c.periocular_embedding is not None:
                return c.periocular_embedding
        return None

    def should_emit(self, interval: float = 2.0) -> bool:
        """Check if we should emit a track-level observation.

        Emits every `interval` seconds while active, or when track ends.
        """
        if not self.crops:
            return False
        if time.time() - self._last_emit_time >= interval:
            return True
        return False

    def mark_emitted(self):
        self._last_emit_time = time.time()
        self._emitted = True

    @property
    def best_crop(self) -> Optional[TrackCrop]:
        return self.crops[0] if self.crops else None


# ─────────────────────────────────────────────────────────────────────────────
#  IR / GRAYSCALE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _is_ir_grayscale(frame: np.ndarray) -> bool:
    """Detect if frame is grayscale/IR (low channel variance).

    Night-mode cameras switch to IR which produces near-grayscale frames.
    Thresholds should be calibrated separately for IR frames.
    """
    if len(frame.shape) < 3:
        return True
    # Sample center region to avoid borders
    h, w = frame.shape[:2]
    roi = frame[h//4:3*h//4, w//4:3*w//4]
    if roi.size == 0:
        return False
    # Compute std of channel differences
    b, g, r = roi[:,:,0].astype(float), roi[:,:,1].astype(float), roi[:,:,2].astype(float)
    diff = np.std(np.abs(b - g)) + np.std(np.abs(g - r))
    return diff < 5.0  # Very low channel variance = grayscale


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION DETECTION (simple frame differencing)
# ─────────────────────────────────────────────────────────────────────────────
class MotionDetector:
    """Simple motion gate using frame differencing.

    When no motion is detected, detection rate drops to save CPU.
    """

    def __init__(self, threshold: float = 5.0, min_area_pct: float = 0.5):
        self._prev_gray = None
        self.threshold = threshold
        self.min_area_pct = min_area_pct

    def has_motion(self, frame: np.ndarray) -> bool:
        """Check if frame has significant motion vs previous frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True  # First frame = assume motion

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pct = (np.count_nonzero(thresh) / thresh.size) * 100
        return motion_pct > self.min_area_pct


# ─────────────────────────────────────────────────────────────────────────────
#  INGEST WORKER (per camera)
# ─────────────────────────────────────────────────────────────────────────────
class IngestWorker:
    """Per-camera ingest worker that runs detection + embedding pipeline.

    Runs in its own thread. Reads frames from CameraSource, runs YOLO
    person detection, then SCRFD face detection on person ROIs, then
    quality assessment + embedding. Stores observations in ObservationStore.
    """

    def __init__(self, camera: CameraSource, embedder: FaceEmbedder,
                 store: ObservationStore, detector=None,
                 detect_fps: float = 5.0, idle_fps: float = 1.0,
                 top_k: int = 5, emit_interval: float = 2.0,
                 track_timeout: float = 5.0):
        self.camera = camera
        self.embedder = embedder
        self.store = store
        self.detector = detector  # PersonDetector (YOLO)
        self.camera_id = camera.camera_id

        # Timing
        self.detect_fps = detect_fps
        self.idle_fps = idle_fps
        self.top_k = top_k
        self.emit_interval = emit_interval
        self.track_timeout = track_timeout

        # State
        self._tracks: Dict[int, TrackState] = {}
        self._motion = MotionDetector()
        self._stopped = False
        self._thread = None

        # Stats
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'embeddings_computed': 0,
            'observations_stored': 0,
            'unusable_logged': 0,
            'persons_without_face': 0,
        }

    def start(self):
        """Start the ingest worker in a background thread."""
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"ingest-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the ingest worker."""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        # Emit any remaining tracks
        self._flush_tracks()

    def _run_loop(self):
        """Main ingest loop: read frames, detect, embed, store."""
        last_detect_time = 0

        while not self._stopped:
            # Check camera health
            if self.camera.health in (CameraHealth.OFFLINE,
                                       CameraHealth.DISABLED):
                time.sleep(1.0)
                continue

            # Read latest frame
            ret, frame, ts = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            # Add to ring buffer
            ring = self.store.get_ring_buffer(self.camera_id)
            ring.add_frame(frame, ts)

            # Adaptive FPS gating
            has_motion = self._motion.has_motion(frame)
            target_interval = 1.0 / (self.detect_fps if has_motion
                                      else self.idle_fps)
            now = time.time()
            if now - last_detect_time < target_interval:
                time.sleep(0.01)
                continue
            last_detect_time = now

            # Run detection pipeline
            self._process_frame(frame, ts)

            # Cleanup stale tracks
            self._cleanup_stale_tracks()

            self.stats['frames_processed'] += 1

    def _process_frame(self, frame: np.ndarray, ts: float):
        """Process one frame through the full pipeline."""
        # Detect IR/grayscale
        is_ir = _is_ir_grayscale(frame)

        # YOLO person detection
        if self.detector is None:
            return

        result = self.detector.detect(frame)

        for det in result.detections:
            track_id = det.track_id
            if track_id < 0:
                continue

            # Extract person ROI
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # Convert to RGB for InsightFace
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # Upscale small ROIs for better face detection
            roi_h, roi_w = roi_rgb.shape[:2]
            scale = 1.0
            if min(roi_h, roi_w) < 160:
                scale = 160.0 / min(roi_h, roi_w)
                roi_rgb = cv2.resize(roi_rgb,
                    (int(roi_w * scale), int(roi_h * scale)),
                    interpolation=cv2.INTER_LINEAR)

            # SCRFD face detection on person ROI
            det_size = (320, 320) if min(roi_rgb.shape[:2]) < 400 else (640, 640)
            faces = self.embedder.detect_faces(roi_rgb, det_size=det_size)

            if not faces:
                # Person present, face unusable — log it
                self._log_unusable(frame, det, track_id, ts, is_ir,
                                   "NO_FACE_DETECTED")
                continue

            # Pick best face by det_score
            face = max(faces, key=lambda f: f.det_score)

            # Map face bbox back to frame coordinates
            if scale != 1.0:
                face.bbox = face.bbox / scale
                if face.kps is not None:
                    face.kps = face.kps / scale
            face_bbox_frame = face.bbox.copy()
            face_bbox_frame[0] += x1
            face_bbox_frame[1] += y1
            face_bbox_frame[2] += x1
            face_bbox_frame[3] += y1

            # Face size
            fw = face.bbox[2] - face.bbox[0]
            fh = face.bbox[3] - face.bbox[1]
            face_size = int(min(fw, fh))

            # Quality assessment
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            face_crop_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
            fx1c = max(0, fx1); fy1c = max(0, fy1)
            fx2c = min(face_crop_bgr.shape[1], fx2)
            fy2c = min(face_crop_bgr.shape[0], fy2)
            face_crop = face_crop_bgr[fy1c:fy2c, fx1c:fx2c]

            if face_crop.size == 0:
                self._log_unusable(frame, det, track_id, ts, is_ir,
                                   "EMPTY_FACE_CROP")
                continue

            qr = assess_quality(face_crop, face._raw_face_obj,
                                frame_w=roi_rgb.shape[1],
                                frame_h=roi_rgb.shape[0])

            # Mask state (feature, not a gate)
            mask_state = "unknown"
            try:
                if self.embedder.is_masked(face, roi_rgb):
                    mask_state = "masked"
                else:
                    mask_state = "unmasked"
            except Exception:
                pass

            # Embedding (always try, even for low quality)
            embedding = face.embedding  # May already be computed by detect_faces
            if embedding is None:
                embedding = self.embedder.embed_face(roi_rgb, face)

            # Periocular embedding
            peri_emb = None
            try:
                peri_emb = self.embedder.embed_periocular(roi_rgb, face)
            except Exception:
                pass

            if embedding is None and not qr.passed:
                # No embedding and bad quality — log unusable
                self._log_unusable(frame, det, track_id, ts, is_ir,
                                   qr.reason)
                continue

            if embedding is None:
                self._log_unusable(frame, det, track_id, ts, is_ir,
                                   "EMBEDDING_FAILED")
                continue

            self.stats['embeddings_computed'] += 1

            # Create track crop
            tc = TrackCrop(
                frame_crop=person_roi,
                face_detection=face,
                quality_score=qr.score,
                embedding=embedding,
                periocular_embedding=peri_emb,
                timestamp=ts,
                bbox=(int(face_bbox_frame[0]), int(face_bbox_frame[1]),
                      int(face_bbox_frame[2]), int(face_bbox_frame[3])),
                det_score=face.det_score,
                face_size_px=face_size,
                yaw=qr.yaw,
                pitch=qr.pitch,
                blur_score=qr.sharpness,
                brightness=qr.brightness,
                mask_state=mask_state,
                is_ir=is_ir,
            )

            # Add to track state
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackState(
                    track_id, self.camera_id, self.top_k
                )
            self._tracks[track_id].add_crop(tc)
            self.stats['faces_detected'] += 1

            # Emit track-level observation periodically
            ts_state = self._tracks[track_id]
            if ts_state.should_emit(self.emit_interval):
                self._emit_observation(ts_state, frame)

    def _log_unusable(self, frame, det, track_id, ts, is_ir, reason):
        """Log 'person present, face unusable' observation."""
        self.stats['unusable_logged'] += 1
        if track_id in self._tracks:
            self._tracks[track_id].unusable_count += 1

        ts_ist = datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
        obs = Observation(
            obs_id=str(uuid.uuid4()),
            camera_id=self.camera_id,
            ts_utc=ts,
            ts_ist=ts_ist,
            track_id=track_id,
            bbox=(det.x1, det.y1, det.x2, det.y2),
            det_score=det.confidence,
            face_size_px=0,
            yaw=0, pitch=0, roll=0,
            blur_score=0, brightness=0, quality_score=0,
            mask_state="unknown",
            is_ir_grayscale=is_ir,
            embedding=np.zeros(self.embedder.embedding_dim, dtype=np.float32),
            periocular_embedding=None,
            crop_path="",
            is_usable=False,
            reject_reason=reason,
        )
        self.store.add_observation(obs)

    def _emit_observation(self, track: TrackState, frame: np.ndarray):
        """Emit a track-level observation to the store."""
        best = track.best_crop
        if best is None:
            return

        # Quality-weighted embedding
        weighted_emb = track.get_weighted_embedding()
        if weighted_emb is None:
            return

        peri_emb = track.get_best_periocular()
        obs_id = str(uuid.uuid4())
        ts_ist = datetime.fromtimestamp(best.timestamp, tz=IST).strftime(
            "%Y-%m-%d %H:%M:%S")

        # Save crop
        crop_path = self.store.save_crop(frame, best.bbox,
                                          self.camera_id, obs_id)

        obs = Observation(
            obs_id=obs_id,
            camera_id=self.camera_id,
            ts_utc=best.timestamp,
            ts_ist=ts_ist,
            track_id=track.track_id,
            bbox=best.bbox,
            det_score=best.det_score,
            face_size_px=best.face_size_px,
            yaw=best.yaw,
            pitch=best.pitch,
            roll=0,
            blur_score=best.blur_score,
            brightness=best.brightness,
            quality_score=best.quality_score,
            mask_state=best.mask_state,
            is_ir_grayscale=best.is_ir,
            embedding=weighted_emb,
            periocular_embedding=peri_emb,
            crop_path=crop_path,
        )
        self.store.add_observation(obs)
        track.mark_emitted()
        self.stats['observations_stored'] += 1

    def _cleanup_stale_tracks(self):
        """Remove tracks not seen for track_timeout seconds."""
        now = time.time()
        stale = [tid for tid, t in self._tracks.items()
                 if now - t.last_seen > self.track_timeout]
        for tid in stale:
            track = self._tracks[tid]
            if not track._emitted and track.crops:
                # Emit final observation for this track
                # Need to use the best crop's frame context
                self._emit_observation_from_crop(track)
            del self._tracks[tid]

    def _emit_observation_from_crop(self, track: TrackState):
        """Emit observation using stored crop data (no frame available)."""
        best = track.best_crop
        if best is None or best.embedding is None:
            return

        weighted_emb = track.get_weighted_embedding()
        if weighted_emb is None:
            return

        peri_emb = track.get_best_periocular()
        obs_id = str(uuid.uuid4())
        ts_ist = datetime.fromtimestamp(best.timestamp, tz=IST).strftime(
            "%Y-%m-%d %H:%M:%S")

        # Save crop from the stored person ROI
        crop_path = ""
        if best.frame_crop is not None and best.frame_crop.size > 0:
            cam_dir = os.path.join(self.store.crops_dir, self.camera_id)
            os.makedirs(cam_dir, exist_ok=True)
            crop_path = os.path.join(cam_dir, f"{obs_id}.jpg")
            cv2.imwrite(crop_path, best.frame_crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        obs = Observation(
            obs_id=obs_id,
            camera_id=self.camera_id,
            ts_utc=best.timestamp,
            ts_ist=ts_ist,
            track_id=track.track_id,
            bbox=best.bbox,
            det_score=best.det_score,
            face_size_px=best.face_size_px,
            yaw=best.yaw,
            pitch=best.pitch,
            roll=0,
            blur_score=best.blur_score,
            brightness=best.brightness,
            quality_score=best.quality_score,
            mask_state=best.mask_state,
            is_ir_grayscale=best.is_ir,
            embedding=weighted_emb,
            periocular_embedding=peri_emb,
            crop_path=crop_path,
        )
        self.store.add_observation(obs)
        track.mark_emitted()
        self.stats['observations_stored'] += 1

    def _flush_tracks(self):
        """Emit all remaining track observations on shutdown."""
        for tid, track in list(self._tracks.items()):
            if not track._emitted and track.crops:
                self._emit_observation_from_crop(track)
        self._tracks.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  INGEST MANAGER (orchestrates all camera workers)
# ─────────────────────────────────────────────────────────────────────────────
class IngestManager:
    """Manages IngestWorker instances for all cameras.

    Each camera gets its own PersonDetector instance because YOLO's
    .track(persist=True) maintains internal tracker state. Sharing one
    detector across cameras would mix track IDs across different video
    streams, producing garbage.

    Usage:
        mgr = IngestManager(camera_manager, embedder, store,
                            detector_kwargs={...})
        mgr.start_all()
        ...
        mgr.stop_all()
    """

    def __init__(self, camera_mgr: CameraManager, embedder: FaceEmbedder,
                 store: ObservationStore, detector=None,
                 detector_kwargs: dict = None,
                 detect_fps: float = 5.0, idle_fps: float = 1.0):
        self.camera_mgr = camera_mgr
        self.embedder = embedder
        self.store = store
        self.workers: Dict[str, IngestWorker] = {}

        for cam_id, cam in camera_mgr.enabled_cameras.items():
            # Create a separate PersonDetector per camera so each
            # has its own YOLO tracker state (persist=True isolation).
            cam_detector = detector  # fallback: shared (single-camera ok)
            if detector_kwargs:
                from detector import PersonDetector
                cam_detector = PersonDetector(**detector_kwargs)

            self.workers[cam_id] = IngestWorker(
                camera=cam,
                embedder=embedder,
                store=store,
                detector=cam_detector,
                detect_fps=detect_fps,
                idle_fps=idle_fps,
            )

    def start_all(self):
        """Start all ingest workers."""
        for cam_id, worker in self.workers.items():
            worker.start()
            print(f"  [Ingest] Started worker for {cam_id}")

    def stop_all(self):
        """Stop all ingest workers gracefully."""
        for cam_id, worker in self.workers.items():
            worker.stop()
        print(f"  [Ingest] All workers stopped")

    def stats(self) -> Dict[str, dict]:
        """Get per-camera ingest statistics."""
        return {cid: w.stats.copy() for cid, w in self.workers.items()}

    def print_stats(self):
        """Print ingest statistics."""
        print(f"\n{'='*60}")
        print(f"  INGEST PIPELINE STATS")
        print(f"{'='*60}")
        for cid, s in self.stats().items():
            print(f"\n  {cid}:")
            print(f"    Frames processed  : {s['frames_processed']}")
            print(f"    Faces detected    : {s['faces_detected']}")
            print(f"    Embeddings        : {s['embeddings_computed']}")
            print(f"    Observations      : {s['observations_stored']}")
            print(f"    Unusable logged   : {s['unusable_logged']}")
        store_stats = self.store.stats()
        print(f"\n  Store: {store_stats['ram_observations']} observations in RAM")
        for cid, rs in store_stats.get('ring_buffers', {}).items():
            print(f"  Ring [{cid}]: {rs['frames']} frames, "
                  f"{rs['span_s']}s span, {rs['memory_mb']}MB")
        print(f"{'='*60}\n")
