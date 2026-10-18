"""
face_embedder.py
================
Swappable face embedding interface for the person search pipeline.

Provides:
  FaceEmbedder      — Abstract interface for face detection + embedding
  InsightFaceEmbedder — Concrete implementation using buffalo_l (SCRFD + ArcFace)

The interface is designed to be replaceable with a commercially-licensed
alternative (see LICENSING.md). Any implementation must provide:
  - detect_faces(img_rgb) → list of FaceDetection
  - embed_face(img_rgb, face) → 512-d L2-normalized embedding
  - embed_periocular(img_rgb, face) → periocular embedding (nullable)
  - is_masked(face, img_rgb) → bool

InsightFace buffalo_l is non-commercial research only. See LICENSING.md.
"""

import os
import threading
import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceDetection:
    """A detected face within an image."""
    bbox: np.ndarray              # [x1, y1, x2, y2]
    det_score: float              # Detection confidence 0-1
    kps: Optional[np.ndarray]     # 5-point landmarks (5,2) or None
    embedding: Optional[np.ndarray] = None  # 512-d L2-normalized or None
    periocular_embedding: Optional[np.ndarray] = None
    mask_state: str = "unknown"   # "masked", "unmasked", "unknown"
    is_synthetic: bool = False    # True if face was inferred from person box
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    _raw_face_obj: object = field(default=None, repr=False)  # Internal


# ─────────────────────────────────────────────────────────────────────────────
#  ABSTRACT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
class FaceEmbedder(ABC):
    """Abstract interface for face detection and embedding.

    Implementations must be thread-safe for concurrent calls from
    multiple camera ingest workers.
    """

    @abstractmethod
    def detect_faces(self, img_rgb: np.ndarray,
                     det_size: Tuple[int, int] = (640, 640)
                     ) -> List[FaceDetection]:
        """Detect faces in an RGB image.

        Args:
            img_rgb: RGB image (H, W, 3) uint8.
            det_size: Detection input resolution.

        Returns:
            List of FaceDetection objects (may be empty).
        """
        ...

    @abstractmethod
    def embed_face(self, img_rgb: np.ndarray,
                   face: FaceDetection) -> Optional[np.ndarray]:
        """Compute full-face embedding for a detected face.

        Args:
            img_rgb: Original RGB image.
            face: FaceDetection from detect_faces().

        Returns:
            512-d L2-normalized embedding, or None on failure.
        """
        ...

    @abstractmethod
    def embed_periocular(self, img_rgb: np.ndarray,
                         face: FaceDetection) -> Optional[np.ndarray]:
        """Compute periocular (upper-face) embedding.

        Returns None if landmarks are insufficient for periocular crop.
        """
        ...

    @abstractmethod
    def is_masked(self, face: FaceDetection,
                  img_rgb: np.ndarray) -> bool:
        """Check if face appears to be wearing a mask.

        NOTE: The MobileNetV2 mask classifier has suspected false-positive
        bias (flagging unmasked faces as masked). Use result as a stored
        feature, not a hard gate.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the face embedding (e.g., 512)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for audit logs."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
#  INSIGHTFACE IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class InsightFaceEmbedder(FaceEmbedder):
    """InsightFace buffalo_l (SCRFD + ArcFace) embedder.

    WARNING: buffalo_l weights are licensed for non-commercial research use
    only. See LICENSING.md for commercial deployment options.
    """

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        import insightface
        from insightface.app import FaceAnalysis

        # Lock for thread safety: InsightFace app.get() and prepare()
        # are not thread-safe. Multiple IngestWorkers share this embedder.
        self._lock = threading.Lock()

        self._app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._det_size = det_size

        # Load mask classifier
        self._mask_sess = None
        mask_path = os.path.join(os.path.dirname(__file__),
                                 "weights", "mask_classifier.onnx")
        if os.path.exists(mask_path):
            import onnxruntime as ort
            self._mask_sess = ort.InferenceSession(
                mask_path, providers=["CPUExecutionProvider"]
            )

        # Detect loaded providers
        providers = "Unknown"
        if self._app.models:
            first_key = list(self._app.models.keys())[0]
            m = self._app.models[first_key]
            if hasattr(m, 'session'):
                providers = m.session.get_providers()
        print(f"  [FaceEmbedder] InsightFace buffalo_l loaded. "
              f"Providers: {providers}")

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "insightface_buffalo_l"

    def detect_faces(self, img_rgb: np.ndarray,
                     det_size: Tuple[int, int] = None
                     ) -> List[FaceDetection]:
        """Detect faces using SCRFD detector. Thread-safe."""
        with self._lock:
            if det_size and det_size != self._det_size:
                self._app.prepare(ctx_id=0, det_size=det_size)
                self._det_size = det_size

            raw_faces = self._app.get(img_rgb)

        results = []
        for f in raw_faces:
            fd = FaceDetection(
                bbox=f.bbox.copy(),
                det_score=float(f.det_score),
                kps=f.kps.copy() if hasattr(f, 'kps') and f.kps is not None else None,
                _raw_face_obj=f,
            )
            # Extract embedding if already computed by .get()
            if hasattr(f, 'embedding') and f.embedding is not None:
                emb = f.embedding / np.linalg.norm(f.embedding)
                fd.embedding = emb
            results.append(fd)
        return results

    def embed_face(self, img_rgb: np.ndarray,
                   face: FaceDetection) -> Optional[np.ndarray]:
        """Get full-face ArcFace embedding. Thread-safe."""
        raw = face._raw_face_obj
        if raw is None:
            return face.embedding  # Already computed

        if face.embedding is not None:
            return face.embedding

        # Force re-compute via recognition model
        try:
            with self._lock:
                self._app.models['recognition'].get(img_rgb, raw)
            if hasattr(raw, 'embedding') and raw.embedding is not None:
                emb = raw.embedding / np.linalg.norm(raw.embedding)
                face.embedding = emb
                return emb
        except Exception:
            pass
        return None

    def embed_periocular(self, img_rgb: np.ndarray,
                         face: FaceDetection) -> Optional[np.ndarray]:
        """Periocular embedding via upper-face masking. Thread-safe."""
        raw = face._raw_face_obj
        if raw is None:
            return None

        try:
            masked_img = self._mask_lower_face(img_rgb, raw)
            if masked_img is None:
                return None
            with self._lock:
                self._app.models['recognition'].get(masked_img, raw)
            if hasattr(raw, 'embedding') and raw.embedding is not None:
                emb = raw.embedding / np.linalg.norm(raw.embedding)
                face.periocular_embedding = emb
                return emb
        except Exception:
            pass
        return None

    def is_masked(self, face: FaceDetection,
                  img_rgb: np.ndarray) -> bool:
        """MobileNetV2 mask classifier. Suspected false-positive bias."""
        if self._mask_sess is None:
            return False
        if face.bbox is None:
            return False
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = y2 - y1, x2 - x1
            pad_y, pad_x = int(h * 0.1), int(w * 0.1)
            cy1 = max(0, y1 - pad_y)
            cy2 = min(img_rgb.shape[0], y2 + pad_y)
            cx1 = max(0, x1 - pad_x)
            cx2 = min(img_rgb.shape[1], x2 + pad_x)
            crop = img_rgb[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return False

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = cv2.resize(crop, (112, 112)).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            inp = np.expand_dims(img, axis=0)

            ort_inputs = {self._mask_sess.get_inputs()[0].name: inp}
            ort_outs = self._mask_sess.run(None, ort_inputs)
            pred = 1.0 / (1.0 + np.exp(-ort_outs[0][0][0]))
            return pred > 0.5
        except Exception:
            return False

    def _mask_lower_face(self, img_rgb, raw_face):
        """Black out lower face for periocular embedding."""
        if not hasattr(raw_face, 'kps') or raw_face.kps is None:
            return None
        kps = raw_face.kps
        if kps.shape[0] < 5:
            return None

        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        mask_y = int(eye_center_y + (nose[1] - eye_center_y) * 0.5)

        if hasattr(raw_face, 'bbox') and raw_face.bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, raw_face.bbox)
        else:
            fx1, fy1, fx2, fy2 = 0, 0, img_rgb.shape[1], img_rgb.shape[0]

        masked = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        masked[max(0, min(h, mask_y)):max(0, min(h, fy2)),
               max(0, min(w, fx1)):max(0, min(w, fx2))] = 0
        return masked
