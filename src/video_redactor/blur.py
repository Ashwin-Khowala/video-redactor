import cv2
import numpy as np

def apply_blur(frame: np.ndarray, bbox: list, padding: int = 10, kernel_size: int = 35) -> None:
    h, w, _ = frame.shape
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    
    min_x = max(0, int(min(x_coords)) - padding)
    max_x = min(w, int(max(x_coords)) + padding)
    min_y = max(0, int(min(y_coords)) - padding)
    max_y = min(h, int(max(y_coords)) + padding)
    
    if max_x > min_x and max_y > min_y:
        roi = frame[min_y:max_y, min_x:max_x]
        if kernel_size % 2 == 0:
            kernel_size += 1
        frame[min_y:max_y, min_x:max_x] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
