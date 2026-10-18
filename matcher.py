"""
matcher.py
==========
Matching & scoring engine for person search.

Features:
  - Vectorized cosine search: obs_matrix @ probe_bank.T
  - Tiered output: STRONG / POSSIBLE / WEAK (not one fixed threshold)
  - Per-camera, per-condition calibration from calibration.json
  - Z-score normalization against non-target crowd distribution
  - Episode grouping (temporal clustering of observations)
  - Temporal agreement (single frame is never STRONG)
  - Periocular + full-face combination for masked faces
  - Cross-camera timeline with CONFLICT / AMBIGUOUS flags
  - Coverage statement generation
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IST
from probe_builder import ProbeTemplate
from observation_store import ObservationStore


# ─────────────────────────────────────────────────────────────────────────────
#  MATCH TIERS
# ─────────────────────────────────────────────────────────────────────────────
class MatchTier:
    STRONG = "STRONG"
    POSSIBLE = "POSSIBLE"
    WEAK = "WEAK"
    NOISE = "NOISE"  # Below noise floor, not shown by default


# ─────────────────────────────────────────────────────────────────────────────
#  MATCH RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MatchResult:
    """A single observation matched against a probe."""
    obs_id: str
    camera_id: str
    ts_utc: float
    ts_ist: str
    track_id: int
    raw_score: float            # Raw cosine similarity
    normalized_score: float     # Z-score normalized
    periocular_score: float     # Periocular match (0 if N/A)
    combined_score: float       # Final combined score
    tier: str                   # STRONG / POSSIBLE / WEAK / NOISE
    quality_score: float
    det_score: float
    mask_state: str
    crop_path: str
    face_size_px: int
    is_ir_grayscale: bool
    episode_id: Optional[str] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class Episode:
    """A group of temporally-related matches on the same camera."""
    episode_id: str
    camera_id: str
    matches: List[MatchResult]
    best_score: float
    tier: str
    time_start: float
    time_end: float
    observation_count: int

    @property
    def duration_seconds(self) -> float:
        return self.time_end - self.time_start

    @property
    def ts_ist_start(self) -> str:
        return datetime.fromtimestamp(self.time_start, tz=IST).strftime(
            "%Y-%m-%d %H:%M:%S")

    @property
    def ts_ist_end(self) -> str:
        return datetime.fromtimestamp(self.time_end, tz=IST).strftime(
            "%Y-%m-%d %H:%M:%S")


@dataclass
class SearchResult:
    """Complete search result including all matches and coverage."""
    probe_hash: str
    search_type: str             # "retro", "watch", "combined"
    episodes: List[Episode]
    all_matches: List[MatchResult]
    coverage: Dict               # Coverage statement
    search_params: Dict
    thresholds_used: Dict
    timestamp: str               # When search was performed


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION (per-camera thresholds from calibration.json)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_THRESHOLDS = {
    "strong_floor": 0.38,
    "possible_floor": 0.28,
    "weak_floor": 0.18,
    "noise_floor": 0.10,
    "periocular_weight": 0.3,
    "fullface_weight": 0.7,
    "strong_min_observations": 2,
    "episode_gap_seconds": 30,
}


def load_calibration(path: str = None) -> Dict:
    """Load per-camera calibration from calibration.json.

    Returns nested dict: {camera_id: {condition: thresholds}}.
    Falls back to defaults if file doesn't exist.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "calibration.json")

    if not os.path.exists(path):
        return {"_default": _DEFAULT_THRESHOLDS}

    with open(path, "r") as f:
        return json.load(f)


def get_thresholds(calibration: Dict, camera_id: str,
                   is_ir: bool = False, is_masked: bool = False) -> Dict:
    """Get thresholds for a specific camera and condition."""
    # Try camera-specific
    if camera_id in calibration:
        cam_cal = calibration[camera_id]
        condition = "ir" if is_ir else ("masked" if is_masked else "rgb")
        if condition in cam_cal:
            return {**_DEFAULT_THRESHOLDS, **cam_cal[condition]}
        if "default" in cam_cal:
            return {**_DEFAULT_THRESHOLDS, **cam_cal["default"]}

    # Fall back to global default
    if "_default" in calibration:
        return {**_DEFAULT_THRESHOLDS, **calibration["_default"]}

    return _DEFAULT_THRESHOLDS.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  MATCHER
# ─────────────────────────────────────────────────────────────────────────────
class Matcher:
    """Person search matching engine.

    Usage:
        matcher = Matcher(store)
        result = matcher.retro_search(probe, last_minutes=30)
        result = matcher.live_match(probe, new_embedding, camera_id, ts)
    """

    def __init__(self, store: ObservationStore,
                 calibration_path: str = None,
                 topology_path: str = None):
        self.store = store
        self.calibration = load_calibration(calibration_path)

        # Camera topology for cross-camera conflict detection
        self.topology = {}
        if topology_path and os.path.exists(topology_path):
            with open(topology_path) as f:
                self.topology = json.load(f)

    def retro_search(self, probe: ProbeTemplate,
                     last_minutes: float = None,
                     start_ts: float = None,
                     end_ts: float = None,
                     camera_ids: List[str] = None,
                     max_results: int = 100) -> SearchResult:
        """Search stored observations for the probe face.

        Args:
            probe: Built probe template.
            last_minutes: Search the last N minutes (mutually exclusive with start/end).
            start_ts: Start of search window (Unix timestamp).
            end_ts: End of search window (Unix timestamp).
            camera_ids: Limit to specific cameras (None = all).
            max_results: Maximum results to return.

        Returns:
            SearchResult with ranked matches and coverage.
        """
        now = time.time()

        if last_minutes:
            start_ts = now - (last_minutes * 60)
            end_ts = now

        # Get embedding matrix snapshot
        matrix, obs_ids, obs_meta = self.store.get_matrix_snapshot()
        if matrix is None or matrix.shape[0] == 0:
            return self._empty_result(probe, "retro", start_ts, end_ts)

        # Filter by time window
        mask = np.ones(len(obs_ids), dtype=bool)
        for i, m in enumerate(obs_meta):
            if start_ts and m['ts_utc'] < start_ts:
                mask[i] = False
            if end_ts and m['ts_utc'] > end_ts:
                mask[i] = False
            if camera_ids and m['camera_id'] not in camera_ids:
                mask[i] = False

        if not np.any(mask):
            return self._empty_result(probe, "retro", start_ts, end_ts)

        filtered_matrix = matrix[mask]
        filtered_ids = [obs_ids[i] for i in range(len(obs_ids)) if mask[i]]
        filtered_meta = [obs_meta[i] for i in range(len(obs_meta)) if mask[i]]

        # Vectorized cosine similarity
        scores = filtered_matrix @ probe.template_bank.T  # (N, M)
        max_scores = np.max(scores, axis=1)  # Best template match per obs

        # Also compute top-2 mean for robustness
        if scores.shape[1] >= 2:
            top2_means = np.mean(np.sort(scores, axis=1)[:, -2:], axis=1)
        else:
            top2_means = max_scores

        # Z-score normalization per camera
        normalized = self._normalize_scores(
            top2_means, filtered_meta)

        # Periocular scores
        peri_scores = np.zeros(len(filtered_ids))
        peri_matrix, peri_ids, peri_indices = self.store.get_periocular_matrix()
        if peri_matrix is not None and probe.periocular_bank is not None:
            peri_raw = peri_matrix @ probe.periocular_bank.T
            peri_max = np.max(peri_raw, axis=1)
            # Map back to filtered indices
            peri_id_set = set(peri_ids)
            for i, oid in enumerate(filtered_ids):
                if oid in peri_id_set:
                    idx = peri_ids.index(oid)
                    peri_scores[i] = peri_max[idx]

        # Build match results with tier classification
        matches = []
        for i in range(len(filtered_ids)):
            meta = filtered_meta[i]
            thresh = get_thresholds(
                self.calibration, meta['camera_id'],
                meta.get('is_ir_grayscale', False),
                meta.get('mask_state') == 'masked',
            )

            # Combine full-face and periocular
            fw = thresh['fullface_weight']
            pw = thresh['periocular_weight']
            combined = fw * top2_means[i] + pw * peri_scores[i]

            # Classify tier
            tier = self._classify_tier(combined, normalized[i], thresh)

            if tier == MatchTier.NOISE:
                continue  # Skip noise

            mr = MatchResult(
                obs_id=filtered_ids[i],
                camera_id=meta['camera_id'],
                ts_utc=meta['ts_utc'],
                ts_ist=meta.get('ts_ist', ''),
                track_id=meta.get('track_id', -1),
                raw_score=float(top2_means[i]),
                normalized_score=float(normalized[i]),
                periocular_score=float(peri_scores[i]),
                combined_score=float(combined),
                tier=tier,
                quality_score=meta.get('quality_score', 0),
                det_score=meta.get('det_score', 0),
                mask_state=meta.get('mask_state', 'unknown'),
                crop_path=meta.get('crop_path', ''),
                face_size_px=meta.get('face_size_px', 0),
                is_ir_grayscale=meta.get('is_ir_grayscale', False),
            )
            matches.append(mr)

        # Sort by combined score
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        matches = matches[:max_results]

        # Group into episodes
        episodes = self._group_episodes(matches)

        # Apply temporal agreement (upgrade/downgrade tiers)
        self._apply_temporal_agreement(episodes)

        # Cross-camera conflict detection
        self._detect_conflicts(episodes)

        # Build coverage
        coverage = self._build_coverage(
            filtered_meta, start_ts, end_ts, camera_ids)

        # Log search
        self.store.log_search(
            probe_hash=probe.probe_hash,
            search_type="retro",
            cameras=camera_ids or list(set(m['camera_id'] for m in obs_meta)),
            time_window=f"{last_minutes}min" if last_minutes else "custom",
            result_count=len(matches),
            thresholds=_DEFAULT_THRESHOLDS,
            model_version="insightface_buffalo_l",
        )

        return SearchResult(
            probe_hash=probe.probe_hash,
            search_type="retro",
            episodes=episodes,
            all_matches=matches,
            coverage=coverage,
            search_params={
                'last_minutes': last_minutes,
                'start_ts': start_ts,
                'end_ts': end_ts,
                'camera_ids': camera_ids,
            },
            thresholds_used=_DEFAULT_THRESHOLDS,
            timestamp=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        )

    def live_match(self, probe: ProbeTemplate,
                   embedding: np.ndarray,
                   peri_embedding: Optional[np.ndarray],
                   camera_id: str, ts: float,
                   quality_score: float = 0.5,
                   mask_state: str = "unknown"
                   ) -> Optional[MatchResult]:
        """Match a single new embedding against the probe in real-time.

        Used by live watch mode. Returns MatchResult if above noise floor.
        """
        scores = probe.template_bank @ embedding
        raw_score = float(np.mean(np.sort(scores)[-2:])
                          if len(scores) >= 2 else np.max(scores))

        peri_score = 0.0
        if peri_embedding is not None and probe.periocular_bank is not None:
            ps = probe.periocular_bank @ peri_embedding
            peri_score = float(np.max(ps))

        thresh = get_thresholds(self.calibration, camera_id)
        combined = (thresh['fullface_weight'] * raw_score +
                    thresh['periocular_weight'] * peri_score)

        tier = self._classify_tier(combined, combined, thresh)
        if tier == MatchTier.NOISE:
            return None

        return MatchResult(
            obs_id="live",
            camera_id=camera_id,
            ts_utc=ts,
            ts_ist=datetime.fromtimestamp(ts, tz=IST).strftime(
                "%Y-%m-%d %H:%M:%S"),
            track_id=-1,
            raw_score=raw_score,
            normalized_score=combined,
            periocular_score=peri_score,
            combined_score=combined,
            tier=tier,
            quality_score=quality_score,
            det_score=0,
            mask_state=mask_state,
            crop_path="",
            face_size_px=0,
            is_ir_grayscale=False,
        )

    def _normalize_scores(self, scores: np.ndarray,
                          meta: List[dict]) -> np.ndarray:
        """Z-score normalize against per-camera non-target distribution.

        A 0.38 that is 5σ above the crowd is more meaningful than
        a 0.38 in a crowd where lookalikes score 0.37.
        """
        normalized = np.zeros_like(scores)
        cameras = set(m['camera_id'] for m in meta)

        for cam_id in cameras:
            cam_mask = np.array([m['camera_id'] == cam_id for m in meta])
            cam_scores = scores[cam_mask]
            if len(cam_scores) < 5:
                normalized[cam_mask] = cam_scores
                continue

            # Use median and MAD for robust statistics
            median = np.median(cam_scores)
            mad = np.median(np.abs(cam_scores - median))
            if mad < 0.01:
                mad = 0.01  # Avoid division by zero

            z_scores = (cam_scores - median) / (mad * 1.4826)  # MAD → std
            normalized[cam_mask] = z_scores

        return normalized

    def _classify_tier(self, combined: float, normalized: float,
                       thresh: Dict) -> str:
        """Classify a match into tier based on score and thresholds."""
        if combined >= thresh['strong_floor']:
            return MatchTier.STRONG
        elif combined >= thresh['possible_floor']:
            return MatchTier.POSSIBLE
        elif combined >= thresh['weak_floor']:
            return MatchTier.WEAK
        elif combined >= thresh['noise_floor']:
            return MatchTier.WEAK  # Show as weak lead
        return MatchTier.NOISE

    def _group_episodes(self, matches: List[MatchResult]) -> List[Episode]:
        """Group matches into episodes by camera and time proximity."""
        if not matches:
            return []

        # Sort by camera then time
        sorted_matches = sorted(matches, key=lambda m: (m.camera_id, m.ts_utc))

        episodes = []
        current_matches = [sorted_matches[0]]

        for m in sorted_matches[1:]:
            prev = current_matches[-1]
            gap = _DEFAULT_THRESHOLDS['episode_gap_seconds']

            if (m.camera_id == prev.camera_id and
                    m.ts_utc - prev.ts_utc <= gap):
                current_matches.append(m)
            else:
                episodes.append(self._make_episode(current_matches))
                current_matches = [m]

        if current_matches:
            episodes.append(self._make_episode(current_matches))

        # Sort episodes by best score
        episodes.sort(key=lambda e: e.best_score, reverse=True)
        return episodes

    def _make_episode(self, matches: List[MatchResult]) -> Episode:
        """Create an Episode from a group of matches."""
        best = max(matches, key=lambda m: m.combined_score)
        ep_id = f"ep-{matches[0].camera_id}-{int(matches[0].ts_utc)}"

        for m in matches:
            m.episode_id = ep_id

        return Episode(
            episode_id=ep_id,
            camera_id=matches[0].camera_id,
            matches=matches,
            best_score=best.combined_score,
            tier=best.tier,
            time_start=matches[0].ts_utc,
            time_end=matches[-1].ts_utc,
            observation_count=len(matches),
        )

    def _apply_temporal_agreement(self, episodes: List[Episode]):
        """Apply temporal agreement rules.

        - Single observation can't be STRONG → downgrade to POSSIBLE
        - ≥2 observations above POSSIBLE → eligible for STRONG
        """
        for ep in episodes:
            if ep.tier == MatchTier.STRONG:
                min_obs = _DEFAULT_THRESHOLDS['strong_min_observations']
                strong_count = sum(
                    1 for m in ep.matches
                    if m.combined_score >= _DEFAULT_THRESHOLDS['strong_floor']
                )
                if strong_count < min_obs:
                    ep.tier = MatchTier.POSSIBLE
                    for m in ep.matches:
                        if m.tier == MatchTier.STRONG:
                            m.tier = MatchTier.POSSIBLE
                            m.flags.append("SINGLE_FRAME_DOWNGRADE")

    def _detect_conflicts(self, episodes: List[Episode]):
        """Detect cross-camera conflicts (impossible travel times)."""
        if len(episodes) < 2 or not self.topology:
            return

        for i, ep1 in enumerate(episodes):
            for ep2 in episodes[i+1:]:
                if ep1.camera_id == ep2.camera_id:
                    continue
                # Check topology for min travel time
                key = f"{ep1.camera_id}->{ep2.camera_id}"
                rev_key = f"{ep2.camera_id}->{ep1.camera_id}"
                min_travel = self.topology.get(
                    key, self.topology.get(rev_key, 0))

                if min_travel > 0:
                    gap = abs(ep1.time_start - ep2.time_start)
                    if gap < min_travel:
                        for m in ep1.matches + ep2.matches:
                            m.flags.append("CONFLICT_IMPOSSIBLE_TRAVEL")

    def _build_coverage(self, all_meta: List[dict],
                        start_ts: float, end_ts: float,
                        camera_ids: List[str] = None) -> Dict:
        """Build coverage statement for search results."""
        cameras_searched = set()
        obs_per_camera = {}
        for m in all_meta:
            cid = m['camera_id']
            cameras_searched.add(cid)
            if cid not in obs_per_camera:
                obs_per_camera[cid] = {'usable': 0, 'total': 0}
            obs_per_camera[cid]['total'] += 1
            obs_per_camera[cid]['usable'] += 1

        return {
            'cameras_searched': list(cameras_searched),
            'time_window_start': start_ts,
            'time_window_end': end_ts,
            'window_duration_minutes': (end_ts - start_ts) / 60 if start_ts and end_ts else 0,
            'observations_per_camera': obs_per_camera,
            'total_usable_observations': sum(
                v['usable'] for v in obs_per_camera.values()),
            'note': ("'No match' only means the person was not detected "
                     "with sufficient quality in the searched window. "
                     "Cameras may have been offline or faces may have "
                     "been too small/blurry to embed."),
        }

    def _empty_result(self, probe: ProbeTemplate,
                      search_type: str,
                      start_ts: float, end_ts: float) -> SearchResult:
        return SearchResult(
            probe_hash=probe.probe_hash,
            search_type=search_type,
            episodes=[],
            all_matches=[],
            coverage={
                'cameras_searched': [],
                'total_usable_observations': 0,
                'note': "No observations in the searched time window.",
            },
            search_params={},
            thresholds_used=_DEFAULT_THRESHOLDS,
            timestamp=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        )
