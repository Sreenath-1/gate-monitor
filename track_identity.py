"""
track_identity.py
=================
Track-level identity aggregation for the gate monitor pipeline (Stage 7).

Each active track maintains a bounded buffer of recognition observations.
Identity decision uses majority vote of individual match results, tie-broken
by a quality-weighted average embedding's match.

This module is deliberately separate from face_id.py (per-frame matching)
to keep per-frame and per-track concerns cleanly decoupled.

Usage:
    from track_identity import TrackIdentityManager

    mgr = TrackIdentityManager(buffer_size=5)
    mgr.update(track_id=5, embedding=emb, quality_score=0.7,
               match_name="Alice", similarity=0.82)
    name, confidence, tier = mgr.get_identity(track_id=5)
"""

import time
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

from config import (
    TRACK_EMBEDDING_BUFFER_K,
    HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD,
    GATE_THIN_MARGIN_THRESHOLD,
)


@dataclass
class Observation:
    """A single recognition observation for a track."""
    embedding: np.ndarray   # 512-dim ArcFace embedding
    quality_score: float    # Quality gate score (0-1)
    match_name: str         # Matched identity name
    similarity: float       # Cosine similarity to the matched identity
    timestamp: float        # time.time() when observed
    method: str = "full"    # "full" or "periocular"
    raw_sim: float = 0.0    # Raw cosine similarity (pre-NAC)
    raw_second_best: float = 0.0  # Runner-up raw sim (for thin-margin check)


class TrackIdentityManager:
    """Manages per-track identity state with embedding fusion.

    Each track maintains a bounded FIFO buffer of Observations. The final
    identity is decided by majority vote, tie-broken by the quality-weighted
    average embedding's cosine similarity against the gallery.

    Confidence = agreement_ratio * mean_similarity_of_agreeing_votes.

    Args:
        buffer_size: Max observations per track (oldest dropped on overflow).
        known_encodings: Dict of {name: {"full": avg_embedding, ...}} from
                         build_known_encodings(). Needed for tie-breaking.
    """

    def __init__(self, buffer_size: int = None,
                 known_encodings: dict = None):
        self._k = buffer_size or TRACK_EMBEDDING_BUFFER_K
        self._known = known_encodings or {}
        # track_id → deque of Observations
        self._buffers: Dict[int, deque] = {}
        # track_id → dict tracking quality gate stats for audit
        self._quality_stats: Dict[int, Dict] = {}

    def update(self, track_id: int, embedding: np.ndarray,
               quality_score: float, match_name: str,
               similarity: float, method: str = "full",
               raw_sim: float = None, raw_second_best: float = None):
        """Push a new recognition observation for a track.

        Args:
            track_id: Persistent track ID.
            embedding: ArcFace embedding vector (512-dim).
            quality_score: Quality gate score (0-1).
            match_name: Name matched by the recognizer.
            similarity: Cosine similarity score.
            method: "full" or "periocular".
            raw_sim: Raw cosine similarity (pre-NAC, for thin-margin).
            raw_second_best: Runner-up raw similarity.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self._k)

        obs = Observation(
            embedding=embedding,
            quality_score=quality_score,
            match_name=match_name,
            similarity=similarity,
            timestamp=time.time(),
            method=method,
            raw_sim=raw_sim if raw_sim is not None else 0.0,
            raw_second_best=raw_second_best if raw_second_best is not None else 0.0,
        )
        self._buffers[track_id].append(obs)

    def record_quality_result(self, track_id: int, passed: bool,
                              reason: str):
        """Record a quality gate result for audit trail.

        Args:
            track_id: Persistent track ID.
            passed: Whether the quality gate passed.
            reason: Reason code (PASS, TOO_BLURRY, etc.)
        """
        if track_id not in self._quality_stats:
            self._quality_stats[track_id] = {
                "frames_considered": 0,
                "frames_passed_quality": 0,
                "rejection_reasons": {},
            }
        stats = self._quality_stats[track_id]
        stats["frames_considered"] += 1
        if passed:
            stats["frames_passed_quality"] += 1
        reason_counts = stats["rejection_reasons"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def get_quality_stats(self, track_id: int) -> dict:
        """Get quality gate statistics for a track (for decision audit).

        Returns:
            Dict with frames_considered, frames_passed_quality,
            rejection_reasons histogram.
        """
        return self._quality_stats.get(track_id, {
            "frames_considered": 0,
            "frames_passed_quality": 0,
            "rejection_reasons": {},
        })

    def get_weighted_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Compute quality-weighted average embedding for a track.

        Weights each embedding by its quality_score so higher-quality
        observations contribute more to the fused embedding.

        Returns:
            Normalized weighted average embedding, or None if no observations.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None

        total_weight = 0.0
        weighted_sum = np.zeros_like(buf[0].embedding)
        for obs in buf:
            w = max(0.01, obs.quality_score)  # Floor to avoid zero weight
            weighted_sum += obs.embedding * w
            total_weight += w

        if total_weight < 1e-9:
            return None

        avg = weighted_sum / total_weight
        norm = np.linalg.norm(avg)
        if norm < 1e-9:
            return None
        return avg / norm

    def get_identity(self, track_id: int) -> Tuple[Optional[str], float, str]:
        """Get the current best identity for a track.

        Uses majority vote of individual match results. Tie-broken by
        quality-weighted average embedding's cosine similarity against
        the gallery.

        Returns:
            (name, confidence, tier):
            - name: Best-guess identity, or None if no observations.
            - confidence: agreement_ratio * mean_similarity_of_agreeing.
            - tier: "high", "uncertain", or "unresolved".
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None, 0.0, "unresolved"

        # Majority vote
        names = [obs.match_name for obs in buf]
        counter = Counter(names)
        most_common_name, vote_count = counter.most_common(1)[0]

        agreement_ratio = vote_count / len(buf)

        # Mean similarity of the agreeing votes
        agreeing_sims = [
            obs.similarity for obs in buf
            if obs.match_name == most_common_name
        ]
        mean_sim = sum(agreeing_sims) / len(agreeing_sims) if agreeing_sims else 0.0

        confidence = agreement_ratio * mean_sim

        # Tie-breaking: if there's a tie, use quality-weighted avg embedding
        if len(counter) > 1:
            top_two = counter.most_common(2)
            if top_two[0][1] == top_two[1][1]:
                # Exact tie — break with weighted embedding
                avg_emb = self.get_weighted_embedding(track_id)
                if avg_emb is not None and self._known:
                    best_tie_name = None
                    best_tie_score = -1.0
                    for name_cand in [top_two[0][0], top_two[1][0]]:
                        if name_cand in self._known:
                            enc = self._known[name_cand]
                            sim = float(np.dot(avg_emb, enc["full"]))
                            if sim > best_tie_score:
                                best_tie_score = sim
                                best_tie_name = name_cand
                    if best_tie_name is not None:
                        most_common_name = best_tie_name
                        confidence = agreement_ratio * best_tie_score

        # Determine tier
        if confidence >= HIGH_CONF_THRESHOLD:
            tier = "high"
        elif confidence >= LOW_CONF_THRESHOLD:
            tier = "uncertain"
        else:
            tier = "unresolved"

        # Thin-margin flagging: if the best identity barely beat the runner-up,
        # downgrade to "uncertain" regardless of raw confidence. This catches
        # ambiguous cases where two enrolled people score nearly identically.
        if tier == "high":
            # Use the latest observation's raw scores for margin check
            latest_agreeing = [
                obs for obs in buf if obs.match_name == most_common_name
            ]
            if latest_agreeing:
                latest = latest_agreeing[-1]
                margin = latest.raw_sim - latest.raw_second_best
                if margin < GATE_THIN_MARGIN_THRESHOLD and latest.raw_second_best > 0:
                    tier = "uncertain"

        return most_common_name, confidence, tier

    def has_observations(self, track_id: int) -> bool:
        """Check if a track has any recognition observations."""
        buf = self._buffers.get(track_id)
        return buf is not None and len(buf) > 0

    def clear_track(self, track_id: int):
        """Remove all state for a track."""
        self._buffers.pop(track_id, None)
        self._quality_stats.pop(track_id, None)

    def clear_stale(self, active_ids: set):
        """Remove state for tracks no longer active."""
        stale = [tid for tid in self._buffers if tid not in active_ids]
        for tid in stale:
            self.clear_track(tid)
