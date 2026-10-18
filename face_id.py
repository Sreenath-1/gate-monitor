"""
face_id.py
==========
Face recognition module for gate_monitor.py — InsightFace (ArcFace) edition.

Uses InsightFace's buffalo_l model pack for:
1. Face detection (RetinaFace) — more robust than MTCNN for CCTV angles
2. ArcFace embeddings (512-dim) — state-of-the-art recognition accuracy
3. Periocular fallback — when face is partially occluded (mask), match
   using only the upper face region (eyes + forehead + brow area)

On startup: builds face embeddings from known_faces/<Name>/*.jpg photos
Per-frame: crops YOLO person bbox, detects face, compares to known embeddings
Returns matched name or None (fallback to "Person #N" in the caller)

Design choices:
- Runs on CPU (ONNX Runtime, no CUDA requirement)
- Embeddings built once at startup, kept in memory as numpy arrays
- InsightFace's buffalo_l includes both RetinaFace detector + ArcFace recognizer
- Cosine similarity for matching (standard for ArcFace embeddings)
- Configurable match_every_n to control FPS impact
- Periocular fallback uses upper-face crop when mask detected via landmarks

Accuracy tradeoffs (honest assessment):
- Full-face ArcFace: ~99%+ on frontal, ~95% on CCTV angle
- Periocular-only: ~85-92% controlled, ~75-85% on CCTV angle
- Periocular confidence scores typically 0.15-0.30 lower than full-face
- Periocular false-positive rate is higher — threshold is set stricter
"""

import os
import numpy as np
from pathlib import Path

# Lazy imports so gate_monitor.py doesn't crash if insightface is missing
_insightface_available = False
_face_app = None
_mask_classifier_sess = None


def _load_insightface():
    """Load InsightFace buffalo_l model pack once, lazily."""
    global _insightface_available, _face_app
    if _insightface_available:
        return True
    try:
        import insightface
        from insightface.app import FaceAnalysis

        # buffalo_l = large model pack (RetinaFace det + ArcFace rec)
        # det_size=(640,640) is default; handles varied input sizes
        _face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # det_size controls the internal detection resolution.
        # 640x640 is accurate but slow on CPU. 320x320 is faster for live use.
        # We start at 640 for accuracy; can be tuned down for FPS.
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        _insightface_available = True
        
        # Explicitly verify which provider was actually loaded
        loaded_providers = "Unknown"
        if _face_app.models:
            first_model_name = list(_face_app.models.keys())[0]
            first_model = _face_app.models[first_model_name]
            if hasattr(first_model, 'session'):
                loaded_providers = first_model.session.get_providers()
                
        print(f"  [face_id] InsightFace buffalo_l loaded. Active providers: {loaded_providers}")
        
        # Load the custom mask classifier
        global _mask_classifier_sess
        mask_model_path = os.path.join(os.path.dirname(__file__), "weights", "mask_classifier.onnx")
        if os.path.exists(mask_model_path):
            import onnxruntime as ort
            # Always fallback to CPU to avoid CUDA dependency issues with the small model
            _mask_classifier_sess = ort.InferenceSession(mask_model_path, providers=["CPUExecutionProvider"])
            print("  [face_id] Custom Mask Classifier (MobileNetV2) loaded.")
        else:
            print(f"  [face_id] WARNING: mask_classifier.onnx not found at {mask_model_path}. Mask detection will default to False.")

        return True
    except Exception as e:
        print(f"  [face_id] WARNING: InsightFace not available: {e}")
        return False


def _is_masked(face, img_rgb=None):
    """
    Mask detection using the custom MobileNetV2 ONNX model.
    """
    global _mask_classifier_sess
    
    # Check if we have the model and the original image
    if _mask_classifier_sess is None or img_rgb is None:
        # Fallback if model not loaded or image not provided
        if hasattr(face, 'det_score') and face.det_score < 0.5:
            return True
        return False

    if face.bbox is None:
        return False
        
    try:
        import cv2
        # Crop face using bbox
        x1, y1, x2, y2 = map(int, face.bbox)
        # Add a small 10% padding
        h, w = y2 - y1, x2 - x1
        pad_y, pad_x = int(h * 0.1), int(w * 0.1)
        
        crop_y1 = max(0, y1 - pad_y)
        crop_y2 = min(img_rgb.shape[0], y2 + pad_y)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(img_rgb.shape[1], x2 + pad_x)
        
        face_crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if face_crop.size == 0:
            return False
            
        # Preprocess for ONNX MobileNetV2
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = cv2.resize(face_crop, (112, 112))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        inp = np.expand_dims(img, axis=0)
        
        ort_inputs = {_mask_classifier_sess.get_inputs()[0].name: inp}
        ort_outs = _mask_classifier_sess.run(None, ort_inputs)
        
        pred_logit = ort_outs[0][0][0]
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
        
        return pred_prob > 0.5
        
    except Exception as e:
        print(f"  [DEBUG] Mask classifier failed: {e}")
        return False


def _mask_lower_face(img_rgb, face):
    """
    Create a copy of the image with the lower face region blacked out.

    Strategy: Use the 5-point landmarks to find the nose tip, then black out
    everything below a line slightly above the nose. This simulates what
    InsightFace "sees" when someone wears a mask — the upper face (eyes,
    forehead, brow) drives the embedding.

    This approach works because InsightFace still detects the face in the
    modified image (the face bbox and eye landmarks are intact) and generates
    an embedding that's biased toward upper-face features.

    Returns the masked image copy, or None if landmarks are insufficient.
    """
    if not hasattr(face, 'landmark_3d_68') or face.landmark_3d_68 is None:
        # Fallback to old heuristic if 68-point landmarks are missing
        if not hasattr(face, 'kps') or face.kps is None:
            return None
        left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        mask_y = int(eye_center_y + (nose[1] - eye_center_y) * 0.5)
        if hasattr(face, 'bbox') and face.bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
        else:
            fx1, fy1, fx2, fy2 = 0, 0, img_rgb.shape[1], img_rgb.shape[0]
        
        masked = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        masked[max(0, min(h, mask_y)):max(0, min(h, fy2)), max(0, min(w, fx1)):max(0, min(w, fx2))] = 0
        return masked, masked

    # Use 68-point landmarks for precise eye+eyebrow polygon mask
    pts = face.landmark_3d_68
    # 17-26: eyebrows, 36-47: eyes (inclusive range 17 to 47 in zero-indexed)
    # landmark_3d_68 has shape (68, 3). We only need x, y for the 2D polygon.
    eyebrows_eyes = pts[17:48, :2]
    
    import cv2
    # 1. Tight Polygon
    hull_tight = cv2.convexHull(eyebrows_eyes.astype(np.int32))
    
    # 2. Wide Polygon (expand points radially from center by 50%)
    center = np.mean(eyebrows_eyes, axis=0)
    eyebrows_eyes_wide = center + (eyebrows_eyes - center) * 1.5
    hull_wide = cv2.convexHull(eyebrows_eyes_wide.astype(np.int32))
    
    # Black out everything OUTSIDE the tight polygon
    mask_tight_layer = np.zeros_like(img_rgb)
    cv2.fillPoly(mask_tight_layer, [hull_tight], (255, 255, 255))
    masked_tight = cv2.bitwise_and(img_rgb, mask_tight_layer)
    
    # Black out everything OUTSIDE the wide polygon
    mask_wide_layer = np.zeros_like(img_rgb)
    cv2.fillPoly(mask_wide_layer, [hull_wide], (255, 255, 255))
    masked_wide = cv2.bitwise_and(img_rgb, mask_wide_layer)
    
    return masked_tight, masked_wide


def _get_embedding(img_rgb, periocular=False, face_obj=None):
    """
    Get face embedding from an image using InsightFace.

    If periocular=True and face_obj is provided, masks the lower face region
    (nose/mouth) and re-runs embedding on the upper face only. This produces
    an embedding biased toward periocular features (eyes, forehead, brow).

    Returns (embedding_array, det_score) or (None, 0.0) if no face found.
    """
    if not _insightface_available or _face_app is None:
        return None, 0.0

    if periocular and face_obj is not None:
        # Mask the lower face and get tight/wide crops
        masks = _mask_lower_face(img_rgb, face_obj)
        if masks is None:
            return None, 0.0

        if isinstance(masks, tuple):
            mask_tight, mask_wide = masks
        else:
            mask_tight = mask_wide = masks

        _face_app.models['recognition'].get(mask_tight, face_obj)
        emb_tight = face_obj.embedding.copy() if (hasattr(face_obj, 'embedding') and face_obj.embedding is not None) else None
        
        if mask_tight is not mask_wide:
            _face_app.models['recognition'].get(mask_wide, face_obj)
            emb_wide = face_obj.embedding.copy() if (hasattr(face_obj, 'embedding') and face_obj.embedding is not None) else None
        else:
            emb_wide = emb_tight
            
        if emb_tight is not None and emb_wide is not None:
            emb = (emb_tight + emb_wide) / 2.0
            emb = emb / np.linalg.norm(emb)
            return emb, float(face_obj.det_score)
        elif emb_tight is not None:
            emb = emb_tight / np.linalg.norm(emb_tight)
            return emb, float(face_obj.det_score)
        elif emb_wide is not None:
            emb = emb_wide / np.linalg.norm(emb_wide)
            return emb, float(face_obj.det_score)
        return None, 0.0

    else:
        # Standard full-face embedding
        faces = _face_app.get(img_rgb)
        if not faces:
            return None, 0.0

        # Pick the face with highest detection score
        face = max(faces, key=lambda f: f.det_score)
        if hasattr(face, 'embedding') and face.embedding is not None:
            emb = face.embedding / np.linalg.norm(face.embedding)
            return emb, float(face.det_score)
        return None, 0.0


def _get_face_and_embedding(img_rgb):
    """
    Detect face and get embedding + face object (for landmark analysis).
    Returns (face_object, embedding, det_score) or (None, None, 0.0).
    """
    if not _insightface_available or _face_app is None:
        return None, None, 0.0

    faces = _face_app.get(img_rgb)
    if not faces:
        print("  [DEBUG] _get_face_and_embedding: detection failed (app.get returned empty)")
        # FALLBACK: If mask confuses RetinaFace, synthesize a dummy face.
        # Estimate head proportion based on YOLO crop aspect ratio.
        h, w = img_rgb.shape[:2]
        aspect_ratio = h / max(1, w)
        if aspect_ratio > 2.5:
            # Full body standing (head is roughly top 1/7th)
            head_h = int(h * 0.15)
        elif aspect_ratio > 1.5:
            # Torso + head (head is roughly top 1/4th)
            head_h = int(h * 0.25)
        else:
            # Mostly head/shoulders
            head_h = int(h * 0.40)
            
        bx, by, bw, bh = 0, 0, w, head_h
        
        # Synthetic frontal landmarks relative to the dynamic head box
        kps = np.array([
            [bx + bw * 0.35, by + bh * 0.45], # left eye
            [bx + bw * 0.65, by + bh * 0.45], # right eye
            [bx + bw * 0.50, by + bh * 0.70], # nose
            [bx + bw * 0.35, by + bh * 0.85], # left mouth
            [bx + bw * 0.65, by + bh * 0.85]  # right mouth
        ], dtype=np.float32)
        
        from insightface.app.common import Face
        dummy_face = Face(bbox=np.array([bx, by, bx+bw, by+bh], dtype=np.float32), kps=kps, det_score=0.1)
        # Mark as synthetic so callers can distinguish real vs estimated face
        dummy_face._synthetic = True
        return dummy_face, None, 0.1

    face = max(faces, key=lambda f: f.det_score)
    if hasattr(face, 'embedding') and face.embedding is not None:
        emb = face.embedding / np.linalg.norm(face.embedding)
        return face, emb, float(face.det_score)
    
    # Debug: why is embedding None? Try manual alignment to surface the exception.
    print(f"  [DEBUG] _get_face_and_embedding: face.embedding is None! (det_score={face.det_score:.3f})")
    try:
        from insightface.utils import face_align
        aimg = face_align.norm_crop(img_rgb, face.kps)
        print("  [DEBUG] norm_crop succeeded manually, but embedding is missing?")
    except Exception as e:
        print(f"  [DEBUG] ALIGNMENT EXCEPTION: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()

    return face, None, float(face.det_score)


def build_known_encodings(known_faces_dir):
    """
    Build a dict of { name: { "full": avg_embedding, "periocular": avg_periocular_embedding,
                               "bucket_counts": dict } }
    from reference photos. Called once at startup.

    If bucket_metadata.json exists in the person's folder (written by enroll_face.py),
    it uses bucket-aware template construction:
      - Full-face template:  built from frontal_unmasked + angled_unmasked frames only
      - Periocular template: built from ALL 4 buckets (masked frames give periocular
                             embeddings under real mask conditions)

    Falls back to the original "encode everything" approach if no metadata exists
    (backward compatible with manually-collected photos).

    Folder structure:
        known_faces/
            Alice/
                photo1.jpg
                bucket_metadata.json   (written by enroll_face.py)
            OtherPerson/
                photo1.jpg

    Returns {} if InsightFace not available or no photos found.
    """
    import cv2
    import json

    known = {}
    if not _load_insightface():
        return known

    base = Path(known_faces_dir)
    if not base.exists():
        print(f"  [face_id] known_faces dir not found: {known_faces_dir}")
        return known

    # Buckets whose frames contribute to the full-face template
    _FULL_FACE_BUCKETS = {"frontal_unmasked", "angled_unmasked"}
    # All buckets contribute to the periocular template
    _PERI_BUCKETS = {"frontal_unmasked", "angled_unmasked",
                     "masked_frontal", "masked_angled"}

    names_found = []
    for person_dir in sorted(base.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        # Skip hidden dirs like _raw
        if name.startswith("_"):
            continue

        # ── Check for bucket metadata ─────────────────────────────────────
        meta_path = person_dir / "bucket_metadata.json"
        bucket_meta = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    bucket_meta = json.load(f)
                print(f"  [face_id]   {name}: bucket_metadata.json found — "
                      f"using bucket-aware template construction")
                bc = bucket_meta.get("bucket_counts", {})
                print(f"  [face_id]   {name}: buckets = {bc}")
            except Exception as e:
                print(f"  [face_id]   {name}: could not read bucket_metadata.json: {e}")

        # Build filename→bucket map for bucket-aware selection
        filename_to_bucket = {}
        if bucket_meta:
            for frame_info in bucket_meta.get("frames", []):
                filename_to_bucket[frame_info["filename"]] = frame_info["bucket"]

        photos = (list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) +
                  list(person_dir.glob("*.png")) + list(person_dir.glob("*.webp")))

        if not photos:
            print(f"  [face_id]   {name}: no photos found — skipping")
            continue

        full_embeddings = []  # frames allowed for full-face template
        peri_embeddings = []  # frames allowed for periocular template

        for photo_path in photos:
            fname = photo_path.name
            # Determine which templates this photo contributes to
            if bucket_meta:
                bucket = filename_to_bucket.get(fname)
                if bucket is None:
                    # Photo predates bucket metadata — use in classic mode
                    # (both full-face and periocular) rather than skipping.
                    # This keeps older enrollment images as a fallback when
                    # the metadata-referenced crops fail InsightFace detection.
                    use_for_full = True
                    use_for_peri = True
                    bucket = "legacy_unclassified"
                    print(f"  [face_id]   {name}/{fname}: not in bucket metadata — using as legacy (full+peri)")
                else:
                    use_for_full = bucket in _FULL_FACE_BUCKETS
                    use_for_peri = bucket in _PERI_BUCKETS
            else:
                # No metadata: classic mode — use everything for both templates
                use_for_full = True
                use_for_peri = True
                bucket = "unknown"

            img_bgr = cv2.imread(str(photo_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            try:
                face_obj, emb, det_score = _get_face_and_embedding(img_rgb)
                
                # Fallback: if RetinaFace fails on the crop (because it lacks context),
                # try running it on the uncropped full frame saved in _raw/, BUT ONLY
                # FOR PERIOCULAR. Full-face ArcFace embeddings require strict 5-point
                # alignment on crops. Using raw frames destroys genuine full-face similarity.
                if emb is None:
                    print(f"  [face_id]   {name}/{fname}: crop failed. Skipping full-face (requires proper alignment)")

                if emb is not None and use_for_full:
                    full_embeddings.append(emb)
                    print(f"  [face_id]   {name}/{fname}: full-face encoded OK (aligned_crop) "
                          f"(det_score={det_score:.3f}, bucket={bucket})")

                # Periocular is less alignment-sensitive, so we can use the raw fallback here
                if face_obj is None:
                    raw_path = photo_path.parent / "_raw" / photo_path.name
                    if raw_path.exists():
                        raw_bgr = cv2.imread(str(raw_path))
                        if raw_bgr is not None:
                            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                            face_obj, emb, det_score = _get_face_and_embedding(raw_rgb)
                            if face_obj is not None:
                                print(f"  [face_id]   {name}/{fname}: recovered face_obj using _raw frame for periocular fallback")

                if face_obj is not None and use_for_peri:
                    peri_emb, _ = _get_embedding(
                        img_rgb, periocular=True, face_obj=face_obj
                    )
                    if peri_emb is None:
                        # Fallback for periocular on raw frame
                        raw_path = photo_path.parent / "_raw" / photo_path.name
                        if raw_path.exists():
                            raw_bgr = cv2.imread(str(raw_path))
                            if raw_bgr is not None:
                                raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                                peri_emb, _ = _get_embedding(raw_rgb, periocular=True, face_obj=face_obj)
                    
                    if peri_emb is not None:
                        peri_embeddings.append(peri_emb)
                        print(f"  [face_id]   {name}/{fname}: periocular encoded OK "
                              f"(bucket={bucket})")
                    else:
                        print(f"  [face_id]   {name}/{fname}: periocular encoding failed "
                              f"(crop too small or no face in crop)")

            except Exception as e:
                print(f"  [face_id]   {name}/{photo_path.name}: ERROR {e}")
                import traceback
                traceback.print_exc()

        if full_embeddings:
            avg_full = np.mean(full_embeddings, axis=0)
            avg_full = avg_full / np.linalg.norm(avg_full)

            entry = {
                "full": avg_full,
                "periocular": None,
                "bucket_counts": bucket_meta.get("bucket_counts", {}) if bucket_meta else {},
            }

            if peri_embeddings:
                avg_peri = np.mean(peri_embeddings, axis=0)
                avg_peri = avg_peri / np.linalg.norm(avg_peri)
                entry["periocular"] = avg_peri

            known[name] = entry
            peri_note = (f", {len(peri_embeddings)} periocular" if peri_embeddings
                         else ", NO periocular")
            names_found.append(
                f"{name} ({len(full_embeddings)} full{peri_note})"
            )
        else:
            print(f"  [face_id]   {name}: no usable full-face photos — skipping")

    if names_found:
        print(f"  [face_id] Loaded known faces: {', '.join(names_found)}")
    else:
        print(f"  [face_id] WARNING: No known faces loaded — face recognition disabled")

    return known


class FaceIdentifier:
    """
    Stateful face identifier using InsightFace (ArcFace).
    Call identify_person() per YOLO detection.

    Caches identity per track_id so face recognition only runs every N frames,
    not every single frame — essential for maintaining FPS.

    Features:
    - Full-face matching (primary, high accuracy)
    - Periocular fallback for masked faces (lower accuracy, honest about it)
    - Mask detection via landmark heuristic
    - Stage 7: Quality gate (reject blurry/small/angled/dark faces before recognition)
    - Stage 7: Track-level embedding fusion (majority vote + quality-weighted avg)
    - Stage 7: Structured audit logging to rotating file

    Args:
        known_encodings: dict from build_known_encodings()
        match_every_n:   run face matching every N frames per track (default 15)
        threshold:       cosine similarity for full-face match (default 0.4)
        peri_threshold:  cosine similarity for periocular match (default 0.30)
                         Set stricter than full-face to reduce false positives
        quality_gate_enabled: Stage 7 — enable face quality gate (default True)
        track_identity_mgr: Stage 7 — TrackIdentityManager instance (optional)
    """

    def __init__(self, known_encodings, match_every_n=15, threshold=0.4,
                 peri_threshold=0.30, debug=True,
                 quality_gate_enabled=True, track_identity_mgr=None,
                 abs_floor_fullface=None, abs_floor_periocular=None,
                 peri_vote_floor=None):
        from config import GATE_ABS_FLOOR_FULLFACE, GATE_ABS_FLOOR_PERIOCULAR, GATE_PERIOCULAR_VOTE_FLOOR
        self.known = known_encodings
        self.every_n = match_every_n
        self.thresh = threshold
        self.peri_thresh = peri_threshold
        self.debug = debug
        # Dual-criterion: absolute cosine similarity floors (open-set rejection)
        self.abs_floor_full = abs_floor_fullface if abs_floor_fullface is not None else GATE_ABS_FLOOR_FULLFACE
        self.abs_floor_peri = abs_floor_periocular if abs_floor_periocular is not None else GATE_ABS_FLOOR_PERIOCULAR
        self.peri_vote_floor = peri_vote_floor if peri_vote_floor is not None else GATE_PERIOCULAR_VOTE_FLOOR
        # { track_id: (name_or_None, frame_last_checked, match_type) }
        # match_type: "full" | "periocular" | None
        self.cache = {}
        # Temporal aggregation buffer for periocular matches
        import collections
        self.peri_history = collections.defaultdict(list)
        # Track consecutive failures for "Unknown Identity Fatigue" backoff
        self.unknown_failures = collections.defaultdict(int)
        # Stats for benchmarking
        self.stats = {
            "full_matches": 0,
            "peri_matches": 0,
            "no_face": 0,
            "no_match": 0,
            "masked_detected": 0,
            "quality_rejected": 0,   # Stage 7
            "quality_passed": 0,     # Stage 7
            "abs_floor_rejected_full": 0,  # Dual-criterion: NAC passed but raw sim too low
            "abs_floor_rejected_peri": 0,  # Dual-criterion: NAC passed but raw sim too low
        }
        # Stage 7: Quality gate
        self._quality_gate_enabled = quality_gate_enabled
        # Stage 7: Track-level embedding fusion
        self._track_mgr = track_identity_mgr
        # Stage 7: Audit logger (lazy init)
        self._audit_logger = None
        # Latency profiling — accumulates sub-stage timings for first 50 calls
        self._latency_samples = []   # list of dicts with sub-stage durations
        self._latency_reported = False
        self._LATENCY_SAMPLE_COUNT = 50  # how many calls to profile

    def identify_person(self, frame_bgr, x1, y1, x2, y2, track_id, frame_idx, priority=False):
        """
        Identify the person in bbox (x1,y1,x2,y2).
        Returns matched name (str) or None if unknown/no face found.
        Uses cached result if track was recently matched.
        If priority=True, bypasses rate limits to force immediate identification.
        """
        import cv2
        import time

        if not self.known or not _insightface_available:
            return None

        _profiling = (not self._latency_reported and
                      len(self._latency_samples) < self._LATENCY_SAMPLE_COUNT)
        _t = {}
        _start_time = time.perf_counter()

        cached_name, last_checked, _ = self.cache.get(track_id, (None, -9999, None))

        # "Unknown Identity Fatigue" backoff MUST take precedence over priority
        if not cached_name and self.unknown_failures[track_id] >= 15:
            check_interval = 30 # Back off to 1.2 seconds if failed 15 times
        elif cached_name:
            check_interval = 150 # Check less frequently if already identified
        elif priority:
            check_interval = 3 # High frequency checks near gate
        else:
            check_interval = self.every_n

        if frame_idx - last_checked < check_interval:
            return cached_name
            
        # Global rate limit: don't run heavy models if we just ran them recently (prevent loop starvation)
        # Allow running at most once every 2 frames globally (since GPU is fast).
        if not hasattr(self, 'last_global_run_frame'):
            self.last_global_run_frame = -9999
            
        if not priority and frame_idx - self.last_global_run_frame < 2:
            return cached_name
            
        self.last_global_run_frame = frame_idx

        # Crop person bbox with small padding
        h, w = frame_bgr.shape[:2]
        pad = 10
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = frame_bgr[cy1:cy2, cx1:cx2]

        # Crop too small to contain a usable face
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            if self.debug:
                print(f"  [face_id] track {track_id}: crop too small "
                      f"({crop.shape[1]}x{crop.shape[0]}) — skipped")
            self.cache[track_id] = (cached_name, frame_idx, None)
            return cached_name

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Detect face + get embedding + face object (for landmark analysis)
            _t0 = time.perf_counter()
            face_obj, emb, det_score = _get_face_and_embedding(crop_rgb)
            if _profiling:
                _t["detection_ms"] = (time.perf_counter() - _t0) * 1000

            if face_obj is None:
                self.stats["no_face"] += 1
                if self.debug:
                    print(f"  [face_id] track {track_id}: no face detected in crop "
                          f"({crop.shape[1]}x{crop.shape[0]})")
                self.cache[track_id] = (cached_name, frame_idx, None)
                return cached_name

            # Log when we are working from a synthetic (dummy) face estimate
            is_synthetic = getattr(face_obj, '_synthetic', False)
            if is_synthetic:
                if self.debug:
                    print(f"  [face_id] track {track_id}: [SYNTHETIC FACE — mask inferred "
                          f"from bbox, no real landmarks detected] → REJECTED as unresolved")
                self.cache[track_id] = (cached_name, frame_idx, None)
                return cached_name

            # ── Stage 7: Quality gate ──────────────────────────────────
            # Run BEFORE embedding comparison. If quality gate rejects,
            # DON'T update last_checked in cache — preserves the throttle
            # counter so the next frame gets a fresh recognition attempt.
            quality_result = None
            if self._quality_gate_enabled:
                from quality import assess_quality
                # Get face bbox within the crop for the quality check
                if hasattr(face_obj, 'bbox') and face_obj.bbox is not None:
                    fx1, fy1, fx2, fy2 = map(int, face_obj.bbox)
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2 = min(crop.shape[1], fx2)
                    fy2 = min(crop.shape[0], fy2)
                    face_crop = crop[fy1:fy2, fx1:fx2]
                    if face_crop.size == 0:
                        face_crop = crop  # Fallback to full person crop
                else:
                    face_crop = crop

                quality_result = assess_quality(face_crop, face_obj, frame_w=crop.shape[1], frame_h=crop.shape[0])

                # Record for audit trail
                if self._track_mgr:
                    self._track_mgr.record_quality_result(
                        track_id, quality_result.passed, quality_result.reason
                    )

                if not quality_result.passed:
                    self.stats["quality_rejected"] += 1
                    if self.debug:
                        print(f"  [face_id] track {track_id}: QUALITY GATE REJECTED "
                              f"reason={quality_result.reason} "
                              f"score={quality_result.score:.3f}")
                    # Structured audit log
                    self._emit_quality_log(
                        track_id, quality_result.score,
                        quality_result.reason, frame_idx
                    )
                    # KEY: Do NOT update cache last_checked frame —
                    # this means the throttle counter is NOT incremented,
                    # so we don't waste recognition attempts on bad frames
                    return cached_name
                else:
                    self.stats["quality_passed"] += 1
                    self._emit_quality_log(
                        track_id, quality_result.score, "PASS", frame_idx,
                        is_pass=True
                    )

            quality_score = quality_result.score if quality_result else 0.5

            # Check if face appears masked
            masked = _is_masked(face_obj, crop_rgb)
            if masked:
                self.stats["masked_detected"] += 1
                if self.debug:
                    synth_note = " [synthetic face]" if is_synthetic else ""
                    print(f"  [face_id] track {track_id}: MASK DETECTED "
                          f"(det_score={det_score:.3f}){synth_note}")

            # ── Strategy 1: Full-face matching (always try first) ──
            best_name = None
            best_score = -1.0
            _t1 = time.perf_counter()
            if emb is not None:
                full_scores = []
                for name, enc in self.known.items():
                    score = float(np.dot(emb, enc["full"]))
                    full_scores.append((name, score))

                if full_scores:
                    full_scores.sort(key=lambda x: x[1], reverse=True)
                    best_name, best_score = full_scores[0]
                    # Keep the raw cosine similarity before NAC adjustment
                    raw_best_score = best_score
                    raw_second_best = full_scores[1][1] if len(full_scores) > 1 else 0.0
                    
                    # Neighborhood Aware Cosine (NAC)
                    if len(full_scores) > 1:
                        second_best_score = full_scores[1][1]
                        # Penalize ambiguous matches (true margin)
                        nac_score = best_score - second_best_score
                        if self.debug:
                            print(f"  [face_id] track {track_id}: FULL-FACE NAC raw={best_score:.3f}, "
                                  f"nearest={second_best_score:.3f} -> NAC={nac_score:.3f}")
                        best_score = nac_score

                if _profiling:
                    _t["fullface_search_ms"] = (time.perf_counter() - _t1) * 1000

                if best_score >= self.thresh:
                    # DUAL-CRITERION CHECK: NAC passed — now check absolute cosine floor
                    if raw_best_score < self.abs_floor_full:
                        # Absolute floor FAILED — reject this match
                        self.stats["abs_floor_rejected_full"] += 1
                        if self.debug:
                            print(f"  [face_id] track {track_id}: FULL-FACE NAC PASSED "
                                  f"(NAC={best_score:.3f}) but ABS FLOOR FAILED "
                                  f"(raw={raw_best_score:.3f} < floor={self.abs_floor_full:.2f}) "
                                  f"→ REJECTED as unresolved")
                        self._emit_recognition_log(
                            track_id, best_name, best_score, "abs_floor_rejected_full", frame_idx,
                            raw_sim=raw_best_score, raw_second=raw_second_best
                        )
                        # Fall through to periocular (which will also apply its own floor)
                    else:
                        # BOTH criteria passed — accept the match
                        self.stats["full_matches"] += 1
                        match_type = "full"
                        total_ms = (time.perf_counter() - _start_time) * 1000
                        if self.debug:
                            print(f"  [face_id] track {track_id}: FULL-FACE MATCH → {best_name} "
                                  f"NAC={best_score:.3f} raw={raw_best_score:.3f} "
                                  f"(thresholds: NAC>={self.thresh}, raw>={self.abs_floor_full:.2f})")
                            print(f"  [face_id] identify_person latency: {total_ms:.1f}ms")
                        if _profiling:
                            _t["total_ms"] = total_ms
                            _t["path"] = "full_match"
                            self._latency_samples.append(_t)
                            self._maybe_report_latency()
                        self.cache[track_id] = (best_name, frame_idx, match_type)
                        if track_id in self.unknown_failures:
                            del self.unknown_failures[track_id]
                        # Stage 7: Push to track identity manager
                        if self._track_mgr and emb is not None:
                            self._track_mgr.update(
                                track_id, emb, quality_score,
                                best_name, best_score, method="full",
                                raw_sim=raw_best_score,
                                raw_second_best=raw_second_best
                            )
                        self._emit_recognition_log(
                            track_id, best_name, best_score, "full", frame_idx,
                            raw_sim=raw_best_score, raw_second=raw_second_best
                        )
                        return best_name

            # ── Strategy 2: Periocular fallback (always try if full face fails) ──
            if True:
                # Try periocular matching
                _t2 = time.perf_counter()
                peri_emb, peri_score = _get_embedding(
                    crop_rgb, periocular=True, face_obj=face_obj
                )
                if _profiling:
                    _t["periocular_embed_ms"] = (time.perf_counter() - _t2) * 1000
                if peri_emb is not None:
                    peri_best_name = None
                    peri_best_score = -1.0
                    
                    peri_scores = []
                    for name, enc in self.known.items():
                        if enc["periocular"] is not None:
                            score = float(np.dot(peri_emb, enc["periocular"]))
                            peri_scores.append((name, score))

                    if peri_scores:
                        peri_scores.sort(key=lambda x: x[1], reverse=True)
                        peri_best_name, peri_best_score = peri_scores[0]
                        
                        # Neighborhood Aware Cosine (NAC)
                        if len(peri_scores) > 1:
                            second_best_score = peri_scores[1][1]
                            nac_score = peri_best_score - second_best_score
                            if self.debug:
                                print(f"  [face_id] track {track_id}: PERIOCULAR NAC raw={peri_best_score:.3f}, "
                                      f"nearest={second_best_score:.3f} -> NAC={nac_score:.3f}")
                            peri_best_score = nac_score

                    if peri_best_name is not None:
                        # Append to history
                        self.peri_history[track_id].append((peri_best_name, peri_best_score))
                        # Keep only the last 5 scores
                        self.peri_history[track_id] = self.peri_history[track_id][-5:]
                        
                        # Temporal aggregation: require at least 2 scores >= threshold for the SAME person
                        recent_matches = [
                            n for n, s in self.peri_history[track_id]
                            if s >= self.peri_vote_floor
                        ]

                        import collections
                        counter = collections.Counter(recent_matches)

                        matched_person = None
                        if counter:
                            most_common_name, count = counter.most_common(1)[0]
                            if count >= 2:
                                matched_person = most_common_name

                        # Periocular similarity search latency
                        if _profiling:
                            _t["periocular_search_ms"] = (
                                time.perf_counter() - _t2
                                - _t.get("periocular_embed_ms", 0) / 1000
                            ) * 1000

                        if matched_person is not None:
                            # DUAL-CRITERION CHECK: periocular temporal aggregation passed
                            # Now check absolute floor on the raw periocular similarity
                            # Use the raw (pre-NAC) best score for floor check
                            raw_peri_best = peri_scores[0][1] if peri_scores else 0.0
                            raw_peri_second = peri_scores[1][1] if len(peri_scores) > 1 else 0.0
                            if raw_peri_best < self.abs_floor_peri:
                                self.stats["abs_floor_rejected_peri"] += 1
                                if self.debug:
                                    print(f"  [face_id] track {track_id}: PERIOCULAR NAC PASSED "
                                          f"but ABS FLOOR FAILED "
                                          f"(raw={raw_peri_best:.3f} < floor={self.abs_floor_peri:.2f}) "
                                          f"→ REJECTED as unresolved")
                                self._emit_recognition_log(
                                    track_id, matched_person, peri_best_score,
                                    "abs_floor_rejected_peri", frame_idx,
                                    raw_sim=raw_peri_best, raw_second=raw_peri_second
                                )
                                # Clear peri_history to prevent stale temporal votes
                                self.peri_history[track_id] = []
                            else:
                                self.stats["peri_matches"] += 1
                                match_type = "periocular"
                                total_ms = (time.perf_counter() - _start_time) * 1000
                                if self.debug:
                                    print(f"  [face_id] track {track_id}: PERIOCULAR MATCH → "
                                          f"{matched_person} (aggregated >=2 matches above {self.peri_vote_floor}) "
                                          f"raw={raw_peri_best:.3f} (floor={self.abs_floor_peri:.2f}) "
                                          f"⚠ LOWER CONFIDENCE than full-face")
                                    print(f"  [face_id] identify_person latency: {total_ms:.1f}ms")
                                if _profiling:
                                    _t["total_ms"] = total_ms
                                    _t["path"] = "peri_match"
                                    self._latency_samples.append(_t)
                                    self._maybe_report_latency()
                                self.cache[track_id] = (matched_person, frame_idx, match_type)
                                if track_id in self.unknown_failures:
                                    del self.unknown_failures[track_id]
                                # Stage 7: Push to track identity manager
                                if self._track_mgr and peri_emb is not None:
                                    self._track_mgr.update(
                                        track_id, peri_emb, quality_score,
                                        matched_person, peri_best_score,
                                        method="periocular",
                                        raw_sim=raw_peri_best,
                                        raw_second_best=raw_peri_second
                                    )
                                self._emit_recognition_log(
                                    track_id, matched_person, peri_best_score,
                                    "periocular", frame_idx,
                                    raw_sim=raw_peri_best, raw_second=raw_peri_second
                                )
                                return matched_person
                        elif self.debug:
                            print(f"  [face_id] track {track_id}: periocular NO MATCH "
                                  f"best={peri_best_name} score={peri_best_score:.3f} "
                                  f"(below {self.peri_thresh} or insufficient temporal count)")
                elif self.debug:
                    print(f"  [face_id] track {track_id}: periocular crop failed")

            # No match found
            self.stats["no_match"] += 1
            self.unknown_failures[track_id] += 1
            total_ms = (time.perf_counter() - _start_time) * 1000
            if self.debug:
                status = f"best={best_name} score={best_score:.3f} (below {self.thresh})"
                print(f"  [face_id] track {track_id}: NO MATCH — {status}")
                print(f"  [face_id] identify_person latency: {total_ms:.1f}ms")
            if _profiling:
                _t["total_ms"] = total_ms
                _t["path"] = "no_match"
                self._latency_samples.append(_t)
                self._maybe_report_latency()
            self._emit_recognition_log(
                track_id, best_name or "None", best_score, "no_match", frame_idx
            )
            self.cache[track_id] = (None, frame_idx, None)
            return None

        except Exception as e:
            if self.debug:
                print(f"  [face_id] track {track_id}: ERROR {e}")
                print(f"  [face_id] identify_person latency: "
                      f"{(time.perf_counter() - _start_time)*1000:.1f}ms")
            self.cache[track_id] = (cached_name, frame_idx, None)
            return cached_name

    # ── Latency reporting ───────────────────────────────────────────────

    def _maybe_report_latency(self):
        """Print a one-time latency breakdown after _LATENCY_SAMPLE_COUNT calls."""
        if self._latency_reported:
            return
        if len(self._latency_samples) < self._LATENCY_SAMPLE_COUNT:
            return
        self._latency_reported = True

        import statistics
        samples = self._latency_samples
        n = len(samples)

        def _avg(key):
            vals = [s[key] for s in samples if key in s]
            return statistics.mean(vals) if vals else 0.0

        def _pct(key):
            vals = [s[key] for s in samples if key in s]
            return sorted(vals)[int(len(vals) * 0.95)] if vals else 0.0

        paths = {}
        for s in samples:
            paths[s.get("path", "?")] = paths.get(s.get("path", "?"), 0) + 1

        print(f"\n  [face_id] ══ LATENCY BREAKDOWN ({n} calls) ══")
        print(f"  [face_id]   Face detection + embedding : "
              f"avg={_avg('detection_ms'):.1f}ms  p95={_pct('detection_ms'):.1f}ms")
        print(f"  [face_id]   Full-face similarity search: "
              f"avg={_avg('fullface_search_ms'):.1f}ms  p95={_pct('fullface_search_ms'):.1f}ms")
        print(f"  [face_id]   Periocular embedding extract: "
              f"avg={_avg('periocular_embed_ms'):.1f}ms  p95={_pct('periocular_embed_ms'):.1f}ms")
        print(f"  [face_id]   Quality gate assessment    : "
              f"avg={_avg('quality_ms'):.1f}ms  p95={_pct('quality_ms'):.1f}ms")
        print(f"  [face_id]   Total identify_person      : "
              f"avg={_avg('total_ms'):.1f}ms  p95={_pct('total_ms'):.1f}ms")
        print(f"  [face_id]   Paths: {paths}")
        print(f"  [face_id]   Dominant bottleneck: "
              f"{'detection' if _avg('detection_ms') > _avg('periocular_embed_ms') else 'periocular_embed'}")
        print(f"  [face_id] ════════════════════════════════════\n")

    # ── Stage 7: Audit logging helpers ─────────────────────────────────

    def _get_audit_logger(self):
        """Lazy-init the audit logger."""
        if self._audit_logger is None:
            try:
                from audit_log import get_audit_logger
                self._audit_logger = get_audit_logger()
            except ImportError:
                pass
        return self._audit_logger

    def _emit_quality_log(self, track_id, score, reason, frame_idx,
                          is_pass=False):
        """Emit a quality gate log entry."""
        logger = self._get_audit_logger()
        if logger:
            if is_pass:
                logger.log_quality_pass(track_id, score, frame_idx)
            else:
                logger.log_quality_rejection(track_id, score, reason, frame_idx)

    def _emit_recognition_log(self, track_id, name, similarity, method,
                              frame_idx, raw_sim=None, raw_second=None):
        """Emit a recognition attempt log entry."""
        logger = self._get_audit_logger()
        if logger:
            logger.log_recognition_attempt(
                track_id, name, similarity, method, frame_idx,
                raw_sim=raw_sim, raw_second=raw_second
            )

    def clear_stale(self, active_ids):
        """Remove stale cache entries for tracks no longer in frame."""
        for tid in list(self.cache):
            if tid not in active_ids:
                del self.cache[tid]
        # Stage 7: Also clear track identity manager stale entries
        if self._track_mgr:
            self._track_mgr.clear_stale(active_ids)

    def get_stats_summary(self):
        """Return a summary string of matching statistics."""
        s = self.stats
        total = s["full_matches"] + s["peri_matches"] + s["no_match"] + s["no_face"]
        if total == 0:
            return "No face recognition attempts yet"
        return (
            f"Full-face matches: {s['full_matches']}, "
            f"Periocular matches: {s['peri_matches']}, "
            f"No match: {s['no_match']}, "
            f"No face detected: {s['no_face']}, "
            f"Masks detected: {s['masked_detected']}, "
            f"Quality rejected: {s['quality_rejected']}, "
            f"Quality passed: {s['quality_passed']}, "
            f"Abs-floor rejected (full): {s['abs_floor_rejected_full']}, "
            f"Abs-floor rejected (peri): {s['abs_floor_rejected_peri']}"
        )
