import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_RTSP_URL
from stream import FrameSource
from face_id import _load_insightface, _face_app
from enroll_quality import assess_enroll_quality

def main():
    if not _load_insightface():
        print("Failed to load insightface")
        return
        
    source = FrameSource(DEFAULT_RTSP_URL, resize_height=0)
    print("Testing yaw... Please turn your head side to side.")
    start_time = time.time()
    
    while time.time() - start_time < 15:
        ret, frame = source.read()
        if not ret:
            time.sleep(0.05)
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = _face_app.get(frame_rgb)
        
        if faces:
            best_face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = map(int, best_face.bbox)
            h_f, w_f = frame.shape[:2]
            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(w_f, x2), min(h_f, y2)
            face_crop_bgr = frame[fy1:fy2, fx1:fx2]
            
            if face_crop_bgr.size > 0:
                enroll_result = assess_enroll_quality(
                    face_crop_bgr, best_face,
                    frontal_yaw_threshold=20.0,
                    frame_w=w_f, frame_h=h_f
                )
                print(f"Yaw: {enroll_result.yaw:.1f}°, Pitch: {enroll_result.pitch:.1f}° | Bucket: {enroll_result.bucket}")
                
    source.release()
    print("Test complete.")

if __name__ == "__main__":
    main()
