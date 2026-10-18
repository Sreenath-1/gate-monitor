"""
audit_log.py
============
Structured debug logging for the gate monitor pipeline (Stage 7).

All per-frame events (quality gate rejections, recognition attempts) are
logged at DEBUG level to a rotating file. Only crossing events are logged
at INFO level. This keeps stdout clean while providing full traceability
for after-the-fact investigation.

Format: single-line structured fields separated by " | " for easy grepping.

Usage:
    from audit_log import get_audit_logger

    logger = get_audit_logger("gate_debug.log")
    logger.log_quality_rejection(track_id=5, score=0.23, reason="TOO_BLURRY")
    logger.log_recognition_attempt(track_id=5, name="Alice", sim=0.72, method="full")
    logger.log_crossing_event(track_id=5, event_type="exit", name="Alice",
                              confidence=0.82, tier="high")
"""

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config import IST


class AuditLogger:
    """Structured logger for gate monitor decision auditing.

    Writes to a rotating file (5MB max, 3 backups). Never prints to stdout —
    all per-frame logs are DEBUG level, crossing events are INFO level.
    """

    def __init__(self, log_path: str):
        self._logger = logging.getLogger("gate_audit")
        self._logger.setLevel(logging.DEBUG)
        # Prevent duplicate handlers if re-initialized
        if not self._logger.handlers:
            handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3,
                encoding="utf-8"
            )
            handler.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(message)s")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            # Don't propagate to root logger (prevents stdout spam)
            self._logger.propagate = False

    def _ts(self) -> str:
        """Current timestamp in IST as compact string."""
        return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def log_quality_rejection(self, track_id: int, score: float,
                              reason: str, frame_idx: int = -1):
        """Log a quality gate rejection (DEBUG level)."""
        self._logger.debug(
            f"{self._ts()} | DEBUG | quality_gate | "
            f"track={track_id} | frame={frame_idx} | "
            f"score={score:.3f} | reason={reason}"
        )

    def log_quality_pass(self, track_id: int, score: float,
                         frame_idx: int = -1):
        """Log a quality gate pass (DEBUG level)."""
        self._logger.debug(
            f"{self._ts()} | DEBUG | quality_gate | "
            f"track={track_id} | frame={frame_idx} | "
            f"score={score:.3f} | reason=PASS"
        )

    def log_recognition_attempt(self, track_id: int, matched_name: str,
                                similarity: float, method: str,
                                frame_idx: int = -1,
                                raw_sim: float = None,
                                raw_second: float = None):
        """Log a face recognition attempt result (DEBUG level)."""
        extra = ""
        if raw_sim is not None:
            extra += f" | raw_sim={raw_sim:.3f}"
        if raw_second is not None:
            extra += f" | raw_second={raw_second:.3f}"
        self._logger.debug(
            f"{self._ts()} | DEBUG | recognition | "
            f"track={track_id} | frame={frame_idx} | "
            f"name={matched_name} | similarity={similarity:.3f} | "
            f"method={method}{extra}"
        )

    def log_crossing_event(self, track_id: int, event_type: str,
                           person_name: str, confidence: float,
                           tier: str, possible_tailgate: bool = False):
        """Log a crossing event (INFO level)."""
        self._logger.info(
            f"{self._ts()} | INFO | crossing | "
            f"track={track_id} | event={event_type} | "
            f"name={person_name} | confidence={confidence:.3f} | "
            f"decision_tier={tier} | tailgate={possible_tailgate}"
        )

    def log_tailgate_detected(self, track_id: int, partner_ids: list,
                              iou: float, consecutive_frames: int):
        """Log a tailgating detection (INFO level)."""
        self._logger.info(
            f"{self._ts()} | INFO | tailgate | "
            f"track={track_id} | partners={partner_ids} | "
            f"iou={iou:.3f} | consecutive_frames={consecutive_frames}"
        )


# Module-level singleton — initialized lazily
_audit_logger = None


def get_audit_logger(log_path: str = None) -> AuditLogger:
    """Get or create the singleton audit logger.

    Args:
        log_path: Path to the debug log file. Only used on first call.
                  Subsequent calls return the existing logger.
    """
    global _audit_logger
    if _audit_logger is None:
        from config import DEBUG_LOG_FILE
        path = log_path or DEBUG_LOG_FILE
        _audit_logger = AuditLogger(path)
    return _audit_logger
