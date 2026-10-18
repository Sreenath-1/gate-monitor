import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_id import _load_insightface, _face_app
from insightface.utils import face_align

def main():
    if not _load_insightface():
        return
        
    img_path = r"known_faces\Alice\Alice_20260927_183106_1.jpg"
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image not found")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print("Running detector...")
    # Det only
    faces = _face_app.models['detection'].detect(img_rgb)
    if faces is None or len(faces[0]) == 0:
        print("No faces detected")
        return
        
    bboxes, kpss = faces
    print(f"Detected {len(bboxes)} faces.")
    kps = kpss[0]
    
    print("Testing norm_crop...")
    try:
        aimg = face_align.norm_crop(img_rgb, kps)
        print("norm_crop succeeded. Shape:", aimg.shape)
    except Exception as e:
        print(f"norm_crop FAILED: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
