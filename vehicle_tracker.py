"""
vehicle_tracker.py
==================
ByteTrack-based multi-object tracker for vehicles.

Wraps supervision.ByteTrack to:
  - Assign persistent track IDs to vehicle detections
  - Maintain per-track state (best plate reads, crop history, vehicle type)
  - Implement temporal fusion (majority-vote plate across track lifetime)
  - Emit completed track events when a vehicle leaves the scene

Architecture:
  Frame detections → ByteTrack → track state update → emit on track loss

Usage:
    tracker = VehicleTracker(fps=25)
    active_tracks, emitted = tracker.update(detections, frame)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from plate_reader import PlateRead
from validators.indian_plate import normalize_plate

logger = logging.getLogger("vehicle_tracker")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


@dataclass
class TrackState:
    """Per-track state maintained across frames."""
    track_id: int
    first_seen: float                      # timestamp
    last_seen: float                       # timestamp
    n_frames: int = 0                      # total frames tracked
    vehicle_type: str = ""                 # majority vote vehicle type
    vehicle_types_seen: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Plate reads (kept sorted by quality)
    plate_reads: List[PlateRead] = field(default_factory=list)
    best_plate: str = ""                   # Current consensus plate text
    plate_confidence: float = 0.0          # Consensus confidence
    plate_valid: bool = False              # Best plate passes validation
    n_valid_reads: int = 0                 # Count of valid plate reads

    # Crops (top-K by quality score)
    best_crops: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    # State
    emitted: bool = False                  # Already emitted as event


@dataclass
class TrackEvent:
    """Emitted when a track is considered complete (vehicle left scene)."""
    track_id: int
    plate_text: str            # Consensus plate (corrected)
    plate_confidence: float
    plate_valid: bool
    vehicle_type: str
    state_code: str
    needs_review: bool
    first_seen: float          # timestamp
    last_seen: float           # timestamp
    duration_s: float
    n_frames: int
    n_plate_reads: int
    n_valid_reads: int
    best_crop: Optional[np.ndarray] = field(default=None, repr=False)
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)


class VehicleTracker:
    """Multi-object vehicle tracker with plate fusion.

    Args:
        fps: Source video FPS (for ByteTrack buffer calculation).
        lost_buffer: Frames to keep a lost track before emitting (default: 30).
        min_consecutive: Minimum consecutive detections before tracking (default: 1).
        quality_topk: Number of best crops to keep per track (default: 8).
        merge_distance: Max edit distance for plate text merging (default: 1).
        emit_min_reads: Minimum plate reads before emitting (default: 3).
        emit_threshold: Minimum plate consensus confidence to emit (default: 0.5).
    """

    def __init__(
        self,
        fps: float = 25,
        lost_buffer: int = 30,
        min_consecutive: int = 1,
        quality_topk: int = 8,
        merge_distance: int = 1,
        emit_min_reads: int = 3,
        emit_threshold: float = 0.5,
    ):
        self.quality_topk = quality_topk
        self.merge_distance = merge_distance
        self.emit_min_reads = emit_min_reads
        self.emit_threshold = emit_threshold

        # ByteTrack initialization
        # Use the wrapped class to avoid deprecation proxy issues
        ByteTrackClass = sv.ByteTrack.wrapped if hasattr(sv.ByteTrack, 'wrapped') else sv.ByteTrack
        self._tracker = ByteTrackClass(
            track_activation_threshold=0.25,
            lost_track_buffer=lost_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=fps,
            minimum_consecutive_frames=min_consecutive,
        )

        # Per-track state: track_id → TrackState
        self._tracks: Dict[int, TrackState] = {}

        # Track IDs seen in previous frame (for detecting lost tracks)
        self._prev_track_ids: set = set()

        logger.info(
            "VehicleTracker initialized: fps=%.0f, lost_buffer=%d, "
            "quality_topk=%d, merge_dist=%d",
            fps, lost_buffer, quality_topk, merge_distance
        )

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        plate_reads_by_det: Optional[Dict[int, List[PlateRead]]] = None,
        vehicle_types: Optional[Dict[int, str]] = None,
    ) -> Tuple[Dict[int, TrackState], List[TrackEvent]]:
        """Update tracker with new detections.

        Args:
            detections: Vehicle detections from VehicleDetector.
            frame: Current BGR frame.
            plate_reads_by_det: Map from detection index → plate reads for that vehicle.
            vehicle_types: Map from detection index → vehicle type string.

        Returns:
            (active_tracks, emitted_events):
              active_tracks: Dict of track_id → TrackState for currently tracked vehicles.
              emitted_events: List of TrackEvent for vehicles that left the scene.
        """
        now = time.time()

        # Run ByteTrack
        tracked = self._tracker.update_with_detections(detections)

        # Get current track IDs
        current_ids = set()
        if tracked.tracker_id is not None:
            current_ids = set(tracked.tracker_id.tolist())

        # Update track states
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                current_ids.add(tid)

                # Create or update track state
                if tid not in self._tracks:
                    self._tracks[tid] = TrackState(
                        track_id=tid,
                        first_seen=now,
                        last_seen=now,
                    )

                track = self._tracks[tid]
                track.last_seen = now
                track.n_frames += 1

                # Vehicle type vote
                if vehicle_types and i in vehicle_types:
                    vtype = vehicle_types[i]
                    track.vehicle_types_seen[vtype] += 1
                    track.vehicle_type = max(
                        track.vehicle_types_seen,
                        key=track.vehicle_types_seen.get
                    )

                # Store crop (scored by sharpness)
                if tracked.xyxy is not None and i < len(tracked.xyxy):
                    x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # Quality score: combination of size and sharpness
                        import cv2
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        area = crop.shape[0] * crop.shape[1]
                        quality = blur_score * np.log1p(area)

                        track.best_crops.append((quality, crop))
                        track.best_crops.sort(key=lambda x: x[0], reverse=True)
                        track.best_crops = track.best_crops[:self.quality_topk]

                # Add plate reads
                if plate_reads_by_det and i in plate_reads_by_det:
                    for read in plate_reads_by_det[i]:
                        track.plate_reads.append(read)
                        if read.plate_valid:
                            track.n_valid_reads += 1

                    # Recompute consensus
                    self._update_plate_consensus(track)

        # Detect lost tracks → emit events
        lost_ids = self._prev_track_ids - current_ids
        emitted = []
        for tid in lost_ids:
            if tid in self._tracks and not self._tracks[tid].emitted:
                track = self._tracks[tid]
                event = self._emit_track(track)
                if event:
                    emitted.append(event)
                    track.emitted = True

        # Clean up old emitted tracks (keep for 60s for potential re-merge)
        cutoff = now - 120
        stale = [tid for tid, t in self._tracks.items()
                 if t.emitted and t.last_seen < cutoff]
        for tid in stale:
            del self._tracks[tid]

        self._prev_track_ids = current_ids

        # Return active (non-emitted) tracks
        active = {tid: t for tid, t in self._tracks.items()
                  if tid in current_ids and not t.emitted}

        return active, emitted

    def _update_plate_consensus(self, track: TrackState):
        """Recompute majority-vote plate text from all reads."""
        valid_reads = [r for r in track.plate_reads if r.plate_valid]

        if not valid_reads:
            # Fall back to all reads
            all_texts = [normalize_plate(r.plate_text) for r in track.plate_reads
                         if r.plate_text]
        else:
            all_texts = [r.corrected_text for r in valid_reads]

        if not all_texts:
            return

        # Group similar texts (within merge_distance)
        groups = []
        for text in all_texts:
            merged = False
            for group in groups:
                if _edit_distance(text, group[0]) <= self.merge_distance:
                    group.append(text)
                    merged = True
                    break
            if not merged:
                groups.append([text])

        # Pick the largest group
        best_group = max(groups, key=len)
        track.best_plate = best_group[0]  # Representative text

        # Confidence = fraction of reads in the winning group
        track.plate_confidence = len(best_group) / len(all_texts)
        track.plate_valid = any(
            r.plate_valid and r.corrected_text == best_group[0]
            for r in track.plate_reads
        )

    def _emit_track(self, track: TrackState) -> Optional[TrackEvent]:
        """Create a TrackEvent from a completed track."""
        duration = track.last_seen - track.first_seen

        # Find state code from best plate read
        state_code = ""
        needs_review = True
        best_valid = [r for r in track.plate_reads
                      if r.plate_valid and r.corrected_text == track.best_plate]
        if best_valid:
            state_code = best_valid[0].state_code
            needs_review = best_valid[0].needs_review

        # Best crop
        best_crop = track.best_crops[0][1] if track.best_crops else None

        # Best plate crop
        plate_crop = None
        plate_reads_sorted = sorted(
            [r for r in track.plate_reads if r.plate_crop is not None],
            key=lambda r: r.plate_width_px * r.blur_score,
            reverse=True,
        )
        if plate_reads_sorted:
            plate_crop = plate_reads_sorted[0].plate_crop

        return TrackEvent(
            track_id=track.track_id,
            plate_text=track.best_plate,
            plate_confidence=track.plate_confidence,
            plate_valid=track.plate_valid,
            vehicle_type=track.vehicle_type,
            state_code=state_code,
            needs_review=needs_review or not track.plate_valid,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            duration_s=duration,
            n_frames=track.n_frames,
            n_plate_reads=len(track.plate_reads),
            n_valid_reads=track.n_valid_reads,
            best_crop=best_crop,
            plate_crop=plate_crop,
        )

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Get a specific track's state."""
        return self._tracks.get(track_id)

    def reset(self):
        """Reset all tracking state."""
        self._tracker.reset()
        self._tracks.clear()
        self._prev_track_ids.clear()
        logger.info("VehicleTracker reset")

    @property
    def n_active(self) -> int:
        """Number of currently active (non-emitted) tracks."""
        return sum(1 for t in self._tracks.values() if not t.emitted)

    @property
    def n_total(self) -> int:
        """Total tracks (including emitted, before cleanup)."""
        return len(self._tracks)
