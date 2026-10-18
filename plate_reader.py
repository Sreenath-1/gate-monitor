"""
plate_reader.py
===============
Licence plate detection + OCR using open-image-models (YOLO-v9-t) and
fast-plate-ocr. Applies Indian plate validation and positional correction.

Architecture:
  Vehicle crop → plate detector → crop plate → OCR → validate → PlateRead

Usage:
    reader = PlateReader()
    reads = reader.read_plates(vehicle_crop)
    reader.close()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("plate_reader")


@dataclass
class PlateRead:
    """Result of reading a single licence plate from a vehicle crop."""
    plate_text: str           # Raw OCR text
    corrected_text: str       # After Indian plate correction
    confidence: float         # OCR confidence (0-1)
    plate_valid: bool         # Passes Indian plate format check
    format_type: str          # standard / bh_series / legacy / invalid
    state_code: str           # State code (e.g. TN, KA, DL)
    needs_review: bool        # Human review needed
    corrected: bool           # Was positional correction applied
    n_corrections: int        # Number of positional corrections
    bbox_in_crop: tuple       # (x1, y1, x2, y2) of plate in vehicle crop
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)
    plate_width_px: int = 0   # Width of plate crop in pixels
    blur_score: float = 0.0   # Laplacian variance (higher = sharper)


class PlateReader:
    """Detect and read licence plates from vehicle crops.

    Combines:
      - open-image-models (YOLO-v9-t) for plate detection
      - fast-plate-ocr for OCR
      - validators.indian_plate for format validation + correction

    Args:
        det_model: Plate detector model name.
        ocr_model: OCR model name from fast-plate-ocr hub.
        min_plate_px: Minimum plate width in pixels to attempt OCR.
        pad_ratio: Padding around detected plate box (fraction of box size).
        device: ONNX Runtime device ('cuda', 'cpu', 'auto').
    """

    def __init__(
        self,
        det_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "global-plates-mobile-vit-v2-model",
        min_plate_px: int = 60,
        pad_ratio: float = 0.12,
        device: str = "auto",
    ):
        self.min_plate_px = min_plate_px
        self.pad_ratio = pad_ratio
        self._det = None
        self._ocr = None

        # Load plate detector (ONNX)
        t0 = time.time()
        try:
            from open_image_models import create_detector
            self._det = create_detector(det_model)
            logger.info("Plate detector loaded in %.1fs: %s", time.time() - t0, det_model)
        except Exception as e:
            logger.error("Plate detector load failed: %s", e)
            raise

        # Load OCR model
        t0 = time.time()
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self._ocr = LicensePlateRecognizer(
                hub_ocr_model=ocr_model,
                device=device,
            )
            logger.info("Plate OCR loaded in %.1fs: %s", time.time() - t0, ocr_model)
        except Exception as e:
            logger.error("Plate OCR load failed: %s", e)
            raise

        # Indian plate validator
        from validators.indian_plate import validate_plate
        self._validate = validate_plate

    def read_plates(
        self,
        vehicle_crop: np.ndarray,
        top_k: int = 3,
    ) -> List[PlateRead]:
        """Detect and read all licence plates in a vehicle crop.

        Args:
            vehicle_crop: BGR numpy array of the vehicle region.
            top_k: Maximum number of plate reads to return (sorted by size).

        Returns:
            List of PlateRead, sorted by plate width (largest first).
            Empty list if no plates detected or all below min_plate_px.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h_crop, w_crop = vehicle_crop.shape[:2]

        # Detect plates
        try:
            detections = self._det.predict(vehicle_crop)
        except Exception as e:
            logger.debug("Plate detection failed: %s", e)
            return []

        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []

        reads = []
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)

            # Pad the plate box
            pw = int((x2 - x1) * self.pad_ratio)
            ph = int((y2 - y1) * self.pad_ratio)
            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(w_crop, x2 + pw)
            y2 = min(h_crop, y2 + ph)

            plate_w = x2 - x1
            plate_h = y2 - y1

            if plate_w < self.min_plate_px or plate_h < 10:
                continue

            plate_crop = vehicle_crop[y1:y2, x1:x2]

            # Blur score (Laplacian variance — higher = sharper)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # OCR
            try:
                preds = self._ocr.run(plate_crop, return_confidence=True)
                if preds and len(preds) > 0:
                    pred = preds[0]
                    raw_text = pred.plate if hasattr(pred, 'plate') else str(pred)

                    # Compute confidence from char_probs
                    conf = 0.0
                    if hasattr(pred, 'char_probs') and pred.char_probs is not None:
                        probs = pred.char_probs
                        if hasattr(probs, '__len__') and len(probs) > 0:
                            conf = float(np.mean([max(p) if hasattr(p, '__iter__') else float(p) for p in probs]))
                        else:
                            conf = float(probs) if probs else 0.0
                    else:
                        conf = 0.5  # Default when no confidence available

                    # Validate and correct
                    validation = self._validate(raw_text)

                    reads.append(PlateRead(
                        plate_text=raw_text,
                        corrected_text=validation.corrected_text,
                        confidence=conf,
                        plate_valid=validation.plate_valid,
                        format_type=validation.format_type,
                        state_code=validation.state_code,
                        needs_review=validation.needs_review,
                        corrected=validation.corrected,
                        n_corrections=validation.n_corrections,
                        bbox_in_crop=(x1, y1, x2, y2),
                        plate_crop=plate_crop,
                        plate_width_px=plate_w,
                        blur_score=blur_score,
                    ))
                else:
                    logger.debug("OCR returned empty for plate %d", i)

            except Exception as e:
                logger.debug("OCR failed for plate %d: %s", i, e)

        # Sort by plate width (largest first) and limit
        reads.sort(key=lambda r: r.plate_width_px, reverse=True)
        return reads[:top_k]

    def read_plates_batch(
        self,
        crops: List[np.ndarray],
    ) -> List[List[PlateRead]]:
        """Run plate reading on a batch of vehicle crops.

        Args:
            crops: List of BGR vehicle crops.

        Returns:
            List of PlateRead lists, one per input crop.
        """
        return [self.read_plates(crop) for crop in crops]

    def close(self):
        """Release detector and OCR resources."""
        self._det = None
        self._ocr = None
        logger.info("PlateReader closed")

    def __del__(self):
        self.close()
