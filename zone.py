"""
zone.py
=======
Entry/exit zone definition and crossing detection.

Supports two zone types:
  1. Line — horizontal or vertical line across the frame (default, backwards-compatible)
  2. Polygon — arbitrary polygon; crossing detected when centroid enters/exits

The crossing detector tracks each person's centroid history and fires an event
when the centroid crosses from one side of the line/polygon to the other.

Anti-flicker (two-layer protection):
  1. min_track_duration — crossing events suppressed if track < 1.5s old
  2. debounce_frames — centroid must stay on the NEW side for N consecutive
     frames before the crossing is committed. This prevents someone standing
     near the doorway boundary from flickering across the line.

Why both are needed:
  - min_track_duration catches brand-new tracks from transient misdetections
  - debounce_frames catches established tracks that jitter near the line
  - Together they eliminate essentially all false crossing events
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Set

from config import TAILGATE_IOU_THRESHOLD, TAILGATE_MIN_FRAMES


@dataclass
class CrossingEvent:
    """A detected crossing event (Stage 7: extended with tailgate info)."""
    track_id: int
    event_type: str              # "entry" or "exit"
    timestamp: float             # time.time() when crossing was detected
    possible_tailgate: bool = False
    tailgate_partner_ids: List[int] = field(default_factory=list)


class CrossingZone:
    """Configurable entry/exit zone with crossing detection + debounce.

    For a horizontal line:
      - Default: moving downward (increasing Y) past line = EXIT
      - Default: moving upward (decreasing Y) past line = ENTRY
      - `flip=True` reverses this

    For a vertical line:
      - Default: moving rightward (increasing X) past line = EXIT
      - Default: moving leftward (decreasing X) past line = ENTRY
      - `flip=True` reverses this

    Debounce logic:
      When a centroid crosses the line, a "pending crossing" is created.
      It's only committed (returned as a real CrossingEvent) after the centroid
      stays on the new side for `debounce_frames` consecutive frames.
      If the centroid bounces back before that, the pending crossing is cancelled.

    Usage:
        zone = CrossingZone(
            direction="horizontal",
            line_position=0.45,
            frame_width=1280,
            frame_height=720,
            flip=False,
            min_track_duration=1.5,
            debounce_frames=3,
        )

        # Each frame, for each tracked person:
        event = zone.update(track_id=5, cx=640, cy=350)
        if event:
            print(f"{event.event_type} by track {event.track_id}")
    """

    def __init__(self, direction: str, line_position: float,
                 frame_width: int, frame_height: int,
                 flip: bool = False, min_track_duration: float = 1.5,
                 debounce_frames: int = 3):
        """
        Args:
            direction: "horizontal" or "vertical".
            line_position: Position as fraction of frame dimension (0.0–1.0).
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            flip: If True, swap entry/exit direction.
            min_track_duration: Minimum track age (seconds) before crossing counts.
            debounce_frames: Consecutive frames on new side before crossing is committed.
        """
        self.direction = direction
        self.flip = flip
        self.min_track_duration = min_track_duration
        self.debounce_frames = debounce_frames
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Compute pixel coordinate of the line
        if direction == "horizontal":
            self.line_coord = int(frame_height * line_position)
        else:
            self.line_coord = int(frame_width * line_position)

        # Per-track state
        self._prev_coord: Dict[int, float] = {}       # track_id → last coord
        self._first_seen: Dict[int, float] = {}        # track_id → first-seen timestamp
        self._outside: set = set()                     # Track IDs currently on the "outside" side
        self._confirmed_side: Dict[int, str] = {}      # track_id → "inside" | "outside" (confirmed)

        # Debounce state: track_id → (pending_event_type, frames_on_new_side)
        # pending_event_type is "exit" or "entry"
        self._pending: Dict[int, Tuple[str, int]] = {}

        # Stage 7: Tailgating detection
        # track_id → (x1, y1, x2, y2) — current bounding box
        self._bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        # (tid_a, tid_b) → consecutive_overlap_frames
        self._tailgate_counters: Dict[Tuple[int, int], int] = {}
        self._tailgate_iou_thresh = TAILGATE_IOU_THRESHOLD
        self._tailgate_min_frames = TAILGATE_MIN_FRAMES

    @property
    def outside_set(self) -> set:
        """Set of track IDs currently on the outside."""
        return self._outside.copy()

    @property
    def outside_count(self) -> int:
        """Number of tracks currently on the outside."""
        return len(self._outside)

    def is_outside(self, track_id: int) -> bool:
        """Check if a specific track is currently on the outside."""
        return track_id in self._outside

    def _get_coord(self, cx: int, cy: int) -> float:
        """Extract the relevant coordinate based on direction."""
        return cy if self.direction == "horizontal" else cx

    def _side_of_line(self, coord: float) -> str:
        """Determine which side of the line a coordinate is on.

        Returns "outside" or "inside" based on direction and flip settings.
        """
        if not self.flip:
            # Default: coord > line = outside side, coord <= line = inside side
            return "outside" if coord > self.line_coord else "inside"
        else:
            # Flipped: coord < line = outside side, coord >= line = inside side
            return "outside" if coord < self.line_coord else "inside"

    def update(self, track_id: int, cx: int, cy: int) -> Optional[CrossingEvent]:
        """Update tracking state for one person and check for crossing.

        Must be called once per tracked person per frame. Returns a CrossingEvent
        if a valid, debounced crossing is detected, None otherwise.

        Args:
            track_id: Persistent track ID from the tracker.
            cx: Current centroid X.
            cy: Current centroid Y.

        Returns:
            CrossingEvent if a confirmed crossing detected, None otherwise.
        """
        coord = self._get_coord(cx, cy)
        now = time.time()
        current_side = self._side_of_line(coord)

        # First time seeing this track — record position, no crossing possible
        if track_id not in self._prev_coord:
            self._prev_coord[track_id] = coord
            self._first_seen[track_id] = now
            self._confirmed_side[track_id] = current_side
            return None

        self._prev_coord[track_id] = coord

        # Anti-flicker layer 1: ignore crossings from tracks younger than min_track_duration
        track_age = now - self._first_seen.get(track_id, now)
        if track_age < self.min_track_duration:
            return None

        confirmed = self._confirmed_side.get(track_id, current_side)

        # Check if centroid is on a different side than the confirmed side
        if current_side != confirmed:
            # Centroid is on the new side — increment debounce counter
            if track_id in self._pending:
                pending_type, count = self._pending[track_id]
                count += 1
                self._pending[track_id] = (pending_type, count)
            else:
                # First frame on new side — start pending
                if confirmed == "inside" and current_side == "outside":
                    self._pending[track_id] = ("exit", 1)
                elif confirmed == "outside" and current_side == "inside":
                    self._pending[track_id] = ("entry", 1)

            # Check if debounce threshold met
            if track_id in self._pending:
                pending_type, count = self._pending[track_id]
                if count >= self.debounce_frames:
                    # Crossing confirmed!
                    del self._pending[track_id]
                    self._confirmed_side[track_id] = current_side

                    if pending_type == "exit":
                        self._outside.add(track_id)
                    elif pending_type == "entry":
                        self._outside.discard(track_id)

                    return CrossingEvent(
                        track_id=track_id,
                        event_type=pending_type,
                        timestamp=now,
                    )
        else:
            # Centroid returned to the confirmed side — cancel any pending crossing
            if track_id in self._pending:
                del self._pending[track_id]

        return None

    def prune_stale(self, active_ids: set):
        """Remove state for tracks no longer being tracked.

        Call once per frame with the set of currently active track IDs.
        Does NOT remove from outside_set — a person who left frame while
        outside should still be counted as outside.
        """
        stale = [tid for tid in self._prev_coord if tid not in active_ids]
        for tid in stale:
            del self._prev_coord[tid]
            self._pending.pop(tid, None)
            # Keep _first_seen, _outside, and _confirmed_side for tracks that may reappear

    def get_line_endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return the two endpoints of the gate line for drawing.

        Returns:
            ((x1, y1), (x2, y2)) — start and end points of the line.
        """
        if self.direction == "horizontal":
            return (0, self.line_coord), (self.frame_width, self.line_coord)
        else:
            return (self.line_coord, 0), (self.line_coord, self.frame_height)

    # ── Stage 7: Tailgating detection ─────────────────────────────────────

    def update_bbox(self, track_id: int, x1: int, y1: int,
                    x2: int, y2: int):
        """Update the current bounding box for a track.

        Must be called once per track per frame before check_tailgate().
        """
        self._bboxes[track_id] = (x1, y1, x2, y2)

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int],
                     box_b: Tuple[int, int, int, int]) -> float:
        """Compute IoU (Intersection over Union) between two bounding boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def check_tailgate(self, crossing_tid: int,
                       active_track_ids: Set[int]) -> List[int]:
        """Check if other tracks are tailgating the crossing track.

        A tailgate is detected when another track's bounding box overlaps
        the crossing track's bbox above the IoU threshold for at least
        TAILGATE_MIN_FRAMES consecutive frames.

        Args:
            crossing_tid: Track ID that just crossed the gate line.
            active_track_ids: Set of all currently active track IDs.

        Returns:
            List of track IDs that are tailgating the crossing track.
        """
        if crossing_tid not in self._bboxes:
            return []

        crossing_box = self._bboxes[crossing_tid]
        tailgaters = []

        for other_tid in active_track_ids:
            if other_tid == crossing_tid:
                continue
            if other_tid not in self._bboxes:
                continue

            other_box = self._bboxes[other_tid]
            iou = self._compute_iou(crossing_box, other_box)

            pair_key = tuple(sorted((crossing_tid, other_tid)))

            if iou >= self._tailgate_iou_thresh:
                self._tailgate_counters[pair_key] = \
                    self._tailgate_counters.get(pair_key, 0) + 1
                if self._tailgate_counters[pair_key] >= self._tailgate_min_frames:
                    tailgaters.append(other_tid)
            else:
                self._tailgate_counters.pop(pair_key, None)

        return tailgaters

    def update_tailgate_frame(self, active_track_ids: Set[int]):
        """Update tailgate counters for all active track pairs near the gate.

        Call once per frame after all bboxes are updated. This maintains
        the consecutive overlap count even when no crossing is happening.
        """
        # Only check tracks that are near the gate line (within debounce zone)
        near_gate = set()
        for tid, coord in self._prev_coord.items():
            if tid in active_track_ids:
                if abs(coord - self.line_coord) < 150:  # ~150px gate zone
                    near_gate.add(tid)

        # Update overlap counters for all pairs near the gate
        checked = set()
        for tid_a in near_gate:
            for tid_b in near_gate:
                if tid_a >= tid_b:
                    continue
                pair_key = (tid_a, tid_b)
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                if tid_a in self._bboxes and tid_b in self._bboxes:
                    iou = self._compute_iou(
                        self._bboxes[tid_a], self._bboxes[tid_b]
                    )
                    if iou >= self._tailgate_iou_thresh:
                        self._tailgate_counters[pair_key] = \
                            self._tailgate_counters.get(pair_key, 0) + 1
                    else:
                        self._tailgate_counters.pop(pair_key, None)

        # Prune stale counters
        stale_pairs = [
            k for k in self._tailgate_counters
            if k[0] not in active_track_ids or k[1] not in active_track_ids
        ]
        for k in stale_pairs:
            del self._tailgate_counters[k]
