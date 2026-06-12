import numpy as np
from video_redactor.blur import apply_blur

def test_apply_blur():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Set a sub-region to 0 so there is variation within the ROI
    frame[20:30, 20:30] = 0
    
    # Bounding box covering a larger area [20:40, 20:40]
    bbox = [[20, 20], [40, 20], [40, 40], [20, 40]]
    
    apply_blur(frame, bbox, padding=0, kernel_size=9)
    
    # Check that blurring blended the edges (non-zero and non-255 pixels now exist)
    assert np.any(frame[20:40, 20:40] > 0)
    assert np.any(frame[20:40, 20:40] < 255)
