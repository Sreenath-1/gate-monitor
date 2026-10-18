# Licensing & Legal Notes

## Face Recognition Models

### InsightFace buffalo_l (SCRFD + ArcFace)

- **License**: Non-commercial research use only
- **Source**: https://github.com/deepinsight/insightface
- **Paper**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **Weights**: Downloaded automatically by InsightFace on first run

> **⚠ IMPORTANT**: The buffalo_l model pack is licensed under a
> non-commercial research license. It **cannot be used in a paid
> commercial deployment** without explicit permission from the
> InsightFace authors (Jia Guo et al., InsightFace team).

### Commercial Alternatives

For commercial/police deployment, replace `InsightFaceEmbedder` in
`face_embedder.py` with one of these commercially-licensed alternatives:

| Provider | License | Notes |
|----------|---------|-------|
| [FaceNet (Google)](https://github.com/davidsandberg/facenet) | Apache 2.0 | Older, lower accuracy than ArcFace |
| [AdaFace](https://github.com/mk-minchul/AdaFace) | MIT | Better on low-quality CCTV |
| [AWS Rekognition](https://aws.amazon.com/rekognition/) | Commercial API | Cloud-based, per-call pricing |
| [Azure Face API](https://azure.microsoft.com/en-us/products/ai-services/ai-vision) | Commercial API | Cloud-based |
| [Luxand FaceSDK](https://luxand.com/) | Commercial SDK | On-premise, per-seat pricing |
| [Cognitec FaceVACS](https://www.cognitec.com/) | Commercial SDK | Police-grade, NIST FRVT tested |
| [NtechLab FindFace](https://ntechlab.com/) | Commercial SDK | Used by Russian law enforcement |

The `FaceEmbedder` interface in `face_embedder.py` is designed for easy
swap — implement `detect_faces()`, `embed_face()`, `embed_periocular()`,
and `is_masked()` for your chosen provider.

### Mask Classifier (MobileNetV2)

- **Architecture**: MobileNetV2 (custom-trained ONNX)
- **License**: Model architecture is Apache 2.0 (Google). Weights are custom.
- **Known Issue**: Suspected high false-positive rate (flags unmasked faces as masked).
  Run `tools/eval_mask_classifier.py` to evaluate on your data.

## YOLO Models

### YOLOv11n (Ultralytics)

- **License**: AGPL-3.0 (open source) or Ultralytics Enterprise License
- **Source**: https://github.com/ultralytics/ultralytics
- **Note**: AGPL requires derivative works to be open-sourced.
  For closed-source commercial use, purchase an Ultralytics Enterprise License.

## Vehicle ANPR Pipeline (--mode vehicles)

### Shipped Components (MIT / Apache-2.0 only)

| Component | Package | License | Source |
|-----------|---------|---------|--------|
| Vehicle detector | `rfdetr` | Apache-2.0 | https://github.com/roboflow/rf-detr |
| Tracker | `supervision` (ByteTrack) | MIT | https://github.com/roboflow/supervision |
| Plate detector | `open-image-models` | Apache-2.0 | https://github.com/ankandrew/open-image-models |
| Plate OCR | `fast-plate-ocr` | Apache-2.0 | https://github.com/ankandrew/fast-plate-ocr |
| CV library | `opencv-python-headless` | Apache-2.0 | https://github.com/opencv/opencv |

### Benchmark-Only (NOT shipped)

| Component | Package | License | Exposure |
|-----------|---------|---------|----------|
| YOLO baseline | `ultralytics` | AGPL-3.0 | Benchmark comparison only. Never imported in shipped code. |

### AGPL / Non-Commercial Exposure — Paused Person/Face Code

> **⚠ The following components exist in the codebase for the paused
> person/face pipeline. They are NOT used by `--mode vehicles` and are
> NOT modified by the vehicle ANPR work.**

| Component | Package | License | Risk |
|-----------|---------|---------|------|
| Person detection + tracking | `ultralytics` (YOLO11n) | AGPL-3.0 | If shipped as closed-source, requires Enterprise License |
| Face recognition | `insightface` (buffalo_l) | Non-commercial research | Cannot be used in paid commercial deployment |
| Face embeddings | ArcFace weights | Non-commercial research | Same as above |


## Legal Considerations (India)

### Digital Personal Data Protection Act, 2023 (DPDP Act)

Face images and biometric embeddings are **personal data** under the DPDP Act.
The following requirements apply:

1. **Consent**: Data principal (person being surveilled) must be informed.
   CCTV systems typically rely on "legitimate use" exemption but this is
   not settled law for face recognition specifically.

2. **Purpose Limitation**: Data collected for "security" cannot be repurposed
   without fresh consent. Searching for a specific person across cameras
   may constitute a new purpose.

3. **Data Minimization**: Store only what is necessary. The system stores
   face embeddings (not reversible to photos) and thumbnails (necessary
   for human review). Configure retention limits appropriately.

4. **Retention**: Default 7-day retention. Configure via `ObservationStore`
   `retention_days` parameter. Do not retain longer than necessary.

5. **Data Principal Rights**: Individuals have the right to request deletion
   of their data under Section 12.

6. **Data Breach**: Notify DPBI (Data Protection Board of India) in case
   of any data breach involving biometric data.

### Recommendations for Deployment

1. **Legal review**: Get explicit sign-off from the client's legal counsel
   before deploying face recognition in any production environment.

2. **Signage**: Post visible notices that face recognition is in use.

3. **Access control**: The web UI binds to `127.0.0.1` by default.
   Do NOT expose to the internet without proper authentication.

4. **Audit trail**: All searches are logged in `search_audit` table with
   probe hash, thresholds, and model version for accountability.

5. **UI language**: System uses "possible match for review" — never
   "identified" or "found". Human verification is always required.

6. **No PII in logs**: Debug logs use track IDs and observation IDs,
   not names or identifying information. Crop paths are the only
   link to visual identity data.

7. **Encryption at rest**: Consider enabling BitLocker or similar for
   the drive containing `observations.db` and `observation_crops/`.

## Disclaimer

This software is a tool for generating **possible leads for human review**.
It does not make identity determinations. False positives and false negatives
are expected — accuracy depends on camera quality, lighting, angle, and
calibration. The operator bears sole responsibility for how search results
are interpreted and acted upon.
