import cv2
import numpy as np
from quality import _estimate_pose, _MODEL_POINTS_5

pts = np.array([
    [100, 100],
    [200, 100],
    [150, 150],
    [120, 200],
    [180, 200]
])

yaw, pitch, roll = _estimate_pose(pts, 640, 480)
print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
