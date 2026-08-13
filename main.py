"""Demo/test script: load an image, run it through a chained PSF pipeline, save result."""
import os

import cv2

from psf import PSFSimulator

INPUT_PATH = r"D:\testPSF\frame_00048_mesh_5220.png"
OUTPUT_PATH = r"D:\testPSF\frame_00048_mesh_5220_psf_eye2.png"

# ordered list of (psf_name, params) applied one after another
# kept subtle: strong values wash the image out into a foggy look
# just blurring is not enough to simulate real-world PSF, so we add a bit of haze and distortion
# PIPELINE = [
#     ("gaussian", {"ksize": (3, 3), "sigma": 0.8}),
#     ("motion", {"length": 1, "angle": 5}),
#     ("atmospheric", {"haze_strength": 0.06}),
# ]
# ordered list of (psf_name, params) applied one after another  fishey outside 
# PIPELINE = [
#     ("fisheye", {"k1": -0.35, "k2": 0.15, "k3": -0.02, "hfov_deg": 80.0}),
# ]

# fishey inside
PIPELINE = [
    ("fisheye", {"k1": 0.35, "k2": -0.15, "k3": 0.02, "hfov_deg": 80.0}),
]

def main() -> None:
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {INPUT_PATH!r}")

    simulator = PSFSimulator()
    result = simulator.apply(img, PIPELINE)

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Saved PSF result to {OUTPUT_PATH!r} using pipeline: {PIPELINE}")


if __name__ == "__main__":
    main()
