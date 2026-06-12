import easyocr
import numpy as np

class OCRProcessor:
    def __init__(self, use_gpu: bool = True):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def detect_text(
        self, 
        frame_rgb: np.ndarray, 
        max_ocr_dim: int, 
        mag_ratio: float, 
        adjust_contrast: bool, 
        min_size: int
    ):
        return self.reader.readtext(
            frame_rgb,
            canvas_size=max_ocr_dim,
            mag_ratio=mag_ratio,
            adjust_contrast=adjust_contrast,
            min_size=min_size
        )
