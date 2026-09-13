
import sys, types
fake = types.ModuleType("ultralytics")
fake.YOLO = object
sys.modules["ultralytics"] = fake

import numpy as np
from segmentation import segment_image, METHODS
from monitoring import validate_rois, ROI

image = np.zeros((30, 30), dtype=np.uint8)
image[:, 15:] = 255

for method in METHODS:
    out = segment_image(image, method)
    assert out.shape == image.shape
    assert set(np.unique(out)).issubset({0, 255})

rois = validate_rois(
    [{"name": "Door", "points": [[0,0],[20,0],[20,20],[0,20]]}],
    30,
    30,
)
assert rois[0].contains((10, 10))
assert not rois[0].contains((25, 25))

print("Day38 segmentation + ROI tests passed.")
