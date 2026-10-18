"""
probe_builder.py
================
Photo → robust probe template bank for person search.

Takes 1..N photos of a target person and builds a template bank for matching:
  - Detect faces; handle 0, 1, or multiple faces
  - Quality assessment with honest warnings
  - Template augmentation: flip, rotations, scale, brightness variants
  - Simulated CCTV degradation to bridge domain gap (ID photo → CCTV frame)
  - Periocular variant for mask/sunglasses cases
  - Multi-photo merge for accuracy boost
  - NO GAN face restoration on matching path (they hallucinate identity)

Probe is logged with SHA-256 hash for audit trail.
"""

import os
import sys
import hashlib
import uuid
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_embedder import FaceEmbedder, FaceDetection


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProbeQualityReport:
    """Quality assessment of a probe photo."""
    face_size_px: int = 0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    blur_score: float = 0.0
    brightness: float = 0.0
    warnings: List[str] = field(default_factory=list)
    reliability: str = "good"  # "good", "moderate", "weak"
    is_profile: bool = False
    is_masked: bool = False
    is_low_res: bool = False


@dataclass
class ProbeTemplate:
    """A built probe template bank ready for matching."""
    probe_id: str                        # UUID
    probe_hash: str                      # SHA-256 of source images
    template_bank: np.ndarray            # (N_templates, 512) L2-normalized
    periocular_bank: Optional[np.ndarray]  # Periocular templates (nullable)
    quality_report: ProbeQualityReport
    face_crops: List[np.ndarray]         # Face crops for display
    source_count: int                    # Number of source photos
    template_count: int                  # Number of templates in bank

    def match_score(self, obs_embedding: np.ndarray,
                    method: str = "max_top2") -> float:
        """Compute match score against this probe.

        Args:
            obs_embedding: 512-d L2-normalized observation embedding.
            method: "max" (max cosine), "max_top2" (mean of top-2 cosines).

        Returns:
            Match score (cosine similarity).
        """
        scores = self.template_bank @ obs_embedding
        if method == "max":
            return float(np.max(scores))
        elif method == "max_top2":
            top = np.sort(scores)[-2:] if len(scores) >= 2 else scores
            return float(np.mean(top))
        return float(np.max(scores))

    def match_periocular(self, peri_embedding: np.ndarray) -> float:
        """Match against periocular bank."""
        if self.periocular_bank is None or peri_embedding is None:
            return 0.0
        scores = self.periocular_bank @ peri_embedding
        return float(np.max(scores))


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class ProbeBuilder:
    """Builds probe template banks from photographs.

    Usage:
        builder = ProbeBuilder(embedder)
        result = builder.build_from_photos(["target.jpg"])
        if result is None:
            print("No face found")
        else:
            print(f"Built {result.template_count} templates")
    """

    def __init__(self, embedder: FaceEmbedder):
        self.embedder = embedder

    def build_from_photos(self, photo_paths: List[str],
                          face_index: int = None
                          ) -> Optional[ProbeTemplate]:
        """Build probe from one or more photos.

        Args:
            photo_paths: List of image file paths.
            face_index: If multiple faces found, select this index (0-based).
                        If None and multiple faces, returns None with error.

        Returns:
            ProbeTemplate or None if no face found.

        Raises:
            ValueError: If multiple faces and face_index not specified.
            FileNotFoundError: If image file not found.
        """
        all_embeddings = []
        all_peri_embeddings = []
        all_crops = []
        quality_report = ProbeQualityReport()

        # Hash all source images
        hasher = hashlib.sha256()

        for path in photo_paths:
            img = self._load_image(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {path}")

            hasher.update(img.tobytes()[:4096])  # Hash first 4KB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = self.embedder.detect_faces(img_rgb)

            if not faces:
                quality_report.warnings.append(
                    f"No face detected in {os.path.basename(path)}. "
                    f"Tips: use a clear frontal photo, good lighting, "
                    f"no heavy occlusion."
                )
                continue

            if len(faces) > 1 and face_index is None and len(photo_paths) == 1:
                # Multiple faces — caller must pick
                raise ValueError(
                    f"Multiple faces ({len(faces)}) detected in "
                    f"{os.path.basename(path)}. Specify face_index (0-based)."
                )

            # Select face
            if face_index is not None and face_index < len(faces):
                face = faces[face_index]
            else:
                face = max(faces, key=lambda f: f.det_score)

            # Get embedding
            emb = face.embedding
            if emb is None:
                emb = self.embedder.embed_face(img_rgb, face)
            if emb is None:
                quality_report.warnings.append(
                    f"Failed to compute embedding for {os.path.basename(path)}"
                )
                continue

            all_embeddings.append(emb)

            # Periocular
            peri = self.embedder.embed_periocular(img_rgb, face)
            if peri is not None:
                all_peri_embeddings.append(peri)

            # Face crop for display
            x1, y1, x2, y2 = map(int, face.bbox)
            pad = int(max(x2-x1, y2-y1) * 0.3)
            cy1 = max(0, y1 - pad)
            cy2 = min(img.shape[0], y2 + pad)
            cx1 = max(0, x1 - pad)
            cx2 = min(img.shape[1], x2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            all_crops.append(crop)

            # Quality metrics from first/best photo
            fw = x2 - x1
            fh = y2 - y1
            quality_report.face_size_px = max(quality_report.face_size_px,
                                               min(fw, fh))

            # Assess quality
            self._assess_probe_quality(crop, face, quality_report)

            # Generate augmented templates from this face
            aug_embs = self._augment_templates(img_rgb, face, emb)
            all_embeddings.extend(aug_embs)

        if not all_embeddings:
            return None

        # Stack and normalize
        bank = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bank = bank / norms

        peri_bank = None
        if all_peri_embeddings:
            peri_bank = np.vstack(all_peri_embeddings).astype(np.float32)
            pn = np.linalg.norm(peri_bank, axis=1, keepdims=True)
            pn[pn == 0] = 1.0
            peri_bank = peri_bank / pn

        probe_hash = hasher.hexdigest()[:16]

        return ProbeTemplate(
            probe_id=str(uuid.uuid4()),
            probe_hash=probe_hash,
            template_bank=bank,
            periocular_bank=peri_bank,
            quality_report=quality_report,
            face_crops=all_crops,
            source_count=len(photo_paths),
            template_count=bank.shape[0],
        )

    def detect_faces_in_photo(self, photo_path: str
                               ) -> List[Tuple[int, np.ndarray, float]]:
        """Detect faces in a photo and return numbered crops.

        Used when multiple faces are found and operator must pick one.

        Returns:
            List of (index, face_crop_bgr, det_score).
        """
        img = self._load_image(photo_path)
        if img is None:
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.embedder.detect_faces(img_rgb)

        results = []
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            pad = int(max(x2-x1, y2-y1) * 0.2)
            cy1 = max(0, y1 - pad)
            cy2 = min(img.shape[0], y2 + pad)
            cx1 = max(0, x1 - pad)
            cx2 = min(img.shape[1], x2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            results.append((i, crop, face.det_score))

        return results

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image handling unicode paths, EXIF rotation, large files."""
        path = str(path)

        # Handle unicode Windows paths
        if not os.path.exists(path):
            return None

        # Read with numpy to handle unicode paths
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = cv2.imread(path)

        if img is None:
            return None

        # Handle EXIF orientation
        img = self._fix_exif_orientation(img, path)

        # Limit size (prevent OOM on huge images)
        max_dim = 2048
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        return img

    def _fix_exif_orientation(self, img: np.ndarray,
                               path: str) -> np.ndarray:
        """Apply EXIF orientation rotation if present."""
        try:
            # Try piexif for EXIF reading (lightweight)
            import piexif
            exif_dict = piexif.load(path)
            orientation = exif_dict.get('0th', {}).get(
                piexif.ImageIFD.Orientation, 1)
            if orientation == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == 6:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass  # No EXIF or piexif not installed — skip
        return img

    def _assess_probe_quality(self, crop_bgr: np.ndarray,
                               face: FaceDetection,
                               report: ProbeQualityReport):
        """Assess probe photo quality and add warnings."""
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Blur
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        report.blur_score = max(report.blur_score, blur)
        if blur < 20:
            report.warnings.append(
                f"Blurry photo (sharpness={blur:.0f}, need >20)")
            report.reliability = "moderate"

        # Face size
        face_px = min(w, h)
        if face_px < 60:
            report.warnings.append(
                f"Face only {face_px}px — upscale won't add real information")
            report.is_low_res = True
            report.reliability = "weak"

        # Brightness
        brightness = float(np.mean(gray))
        report.brightness = brightness
        if brightness < 40 or brightness > 220:
            report.warnings.append(
                f"Extreme brightness ({brightness:.0f}) — expect weaker matches")

        # Yaw/pitch
        if face.kps is not None and face.kps.shape[0] >= 5:
            from quality import _estimate_pose
            yaw, pitch, _ = _estimate_pose(face.kps, w, h)
            report.yaw_deg = yaw
            report.pitch_deg = pitch
            if abs(yaw) > 30:
                report.warnings.append(
                    f"Profile photo (yaw={yaw:.0f}°) — expect weak matches")
                report.is_profile = True
                report.reliability = "weak" if abs(yaw) > 45 else "moderate"

    def _augment_templates(self, img_rgb: np.ndarray,
                            face: FaceDetection,
                            original_emb: np.ndarray) -> List[np.ndarray]:
        """Generate augmented template embeddings.

        Augmentations:
          1. Horizontal flip
          2. Small rotations (±8°)
          3. Brightness/contrast variants
          4. Simulated CCTV degradation (downscale→JPEG→blur→noise)
        """
        augmented = []

        # 1. Horizontal flip
        flipped = cv2.flip(img_rgb, 1)
        faces = self.embedder.detect_faces(flipped)
        if faces:
            f = max(faces, key=lambda x: x.det_score)
            emb = f.embedding
            if emb is None:
                emb = self.embedder.embed_face(flipped, f)
            if emb is not None:
                augmented.append(emb)

        # 2. Small rotations (±8°)
        for angle in [-8, 8]:
            rotated = self._rotate_image(img_rgb, angle)
            faces = self.embedder.detect_faces(rotated)
            if faces:
                f = max(faces, key=lambda x: x.det_score)
                emb = f.embedding
                if emb is None:
                    emb = self.embedder.embed_face(rotated, f)
                if emb is not None:
                    augmented.append(emb)

        # 3. Brightness/contrast variants
        for alpha, beta in [(0.85, -10), (1.15, 10)]:
            adjusted = np.clip(img_rgb * alpha + beta, 0, 255).astype(np.uint8)
            faces = self.embedder.detect_faces(adjusted)
            if faces:
                f = max(faces, key=lambda x: x.det_score)
                emb = f.embedding
                if emb is None:
                    emb = self.embedder.embed_face(adjusted, f)
                if emb is not None:
                    augmented.append(emb)

        # 4. Simulated CCTV degradation
        degraded = self._simulate_cctv(img_rgb)
        faces = self.embedder.detect_faces(degraded)
        if faces:
            f = max(faces, key=lambda x: x.det_score)
            emb = f.embedding
            if emb is None:
                emb = self.embedder.embed_face(degraded, f)
            if emb is not None:
                augmented.append(emb)

        return augmented

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees around center."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _simulate_cctv(self, img_rgb: np.ndarray) -> np.ndarray:
        """Simulate CCTV-quality degradation of an ID photo.

        Applies: downscale → JPEG compress → blur → noise
        to narrow the domain gap between ID photos and CCTV frames.
        """
        h, w = img_rgb.shape[:2]

        # Downscale to ~25% then back up (simulates low CCTV resolution)
        small = cv2.resize(img_rgb, (max(w//4, 64), max(h//4, 64)))
        upscaled = cv2.resize(small, (w, h))

        # JPEG compression artifacts
        _, jpeg = cv2.imencode('.jpg', upscaled,
                                [cv2.IMWRITE_JPEG_QUALITY, 40])
        upscaled = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if upscaled.shape[2] == 3 and len(img_rgb.shape) == 3:
            # Convert back to RGB if needed
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)

        # Gaussian blur (motion/focus blur)
        upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0.8)

        # Light noise
        noise = np.random.normal(0, 8, upscaled.shape).astype(np.float32)
        upscaled = np.clip(upscaled.astype(np.float32) + noise, 0, 255
                           ).astype(np.uint8)

        return upscaled
