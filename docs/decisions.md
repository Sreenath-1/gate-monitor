# Vehicle ANPR: decision log

D1. Build the vehicle mode inside the existing gate monitor to reuse stream, camera, zone and logging code.
D2. RF-DETR Nano is the vehicle detector (Apache-2.0; 29.6 FPS and 173 MB measured on the dev GPU with synthetic frames). Ultralytics YOLO is used only as a benchmark baseline for vehicles (AGPL-3.0).
D3. ByteTrack from the Roboflow trackers library tracks vehicles, independent of the detector.
D4. Plates are detected on vehicle crops cut from the full-resolution frame, because plates are too small in a downscaled whole frame.
D5. The global fast-plate-ocr model is used first. Fine-tune on Indian plates only if measurement shows it is needed.
D6. Character correction is positional only, capped at two edits, and always flagged. Unresolved reads go to a human, never forced valid.
D7. Cross-frame voting must beat the best single frame in an ablation, because published work shows fusion can degrade results when tracks fragment.
D8. Every clip carries a source label (phone, public, simulated, cctv). Client accuracy statements use cctv data only.
D9. Testing follows a ladder: file replay, simulated RTSP, phone RTSP, then the real camera.
D10. The development laptop is for development and pilot only; production hardware is sized from measured numbers.
D11. Operator corrections feed a training export for OCR fine-tuning.
D12. All data stays local with configurable retention, because plates are personal data.
