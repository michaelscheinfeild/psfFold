"""Simulate camera Point Spread Function (PSF) degradations on images.

Provides Gaussian, Airy disk (diffraction), motion blur, atmospheric haze
and fisheye distortion, chainable via PSFSimulator.apply(), so sharp
synthetic RGB/IR renders can be made to look like real camera output.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.special import j1


class PSFSimulator:
    """Applies one or more camera PSF effects to an 8-bit 3-channel BGR image."""

    def apply_gaussian(
        self, img: np.ndarray, ksize: Tuple[int, int] = (7, 7), sigma: float = 1.8
    ) -> np.ndarray:
        """Approximate lens softness + sensor diffusion with a Gaussian blur."""
        return cv2.GaussianBlur(img, ksize, sigmaX=sigma)

    def apply_airy(
        self, img: np.ndarray, radius_px: float = 3.0, kernel_size: int = 15
    ) -> np.ndarray:
        """Diffraction-limited blur using an Airy disk (jinc) kernel."""
        kernel = self._airy_kernel(radius_px, kernel_size)
        return cv2.filter2D(img, -1, kernel)

    def apply_motion(
        self, img: np.ndarray, length: int = 15, angle: float = 0.0
    ) -> np.ndarray:
        """Directional blur simulating drone/camera motion during exposure."""
        kernel = self._motion_kernel(length, angle)
        return cv2.filter2D(img, -1, kernel)

    def apply_atmospheric(
        self,
        img: np.ndarray,
        haze_strength: float = 0.3,
        airlight: Tuple[int, int, int] = (230, 230, 230),
    ) -> np.ndarray:
        """Blend in airlight to simulate atmospheric haze/scattering."""
        airlight_arr = np.array(airlight, dtype=np.float32)
        out = img.astype(np.float32) * (1.0 - haze_strength) + airlight_arr * haze_strength
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_fisheye(
        self,
        img: np.ndarray,
        k1: float = -0.35,
        k2: float = 0.15,
        p1: float = 0.0,
        p2: float = 0.0,
        k3: float = -0.02,
        hfov_deg: float = 80.0,
    ) -> np.ndarray:
        """Barrel/pincushion distortion via a pinhole camera + polynomial radial model.

        Camera focal length is derived from the desired horizontal FOV rather than
        a fixed value, so the distortion strength scales correctly with lens angle.
        Note: true fisheye lenses beyond ~120-140 deg HFOV are better modeled with
        an equidistant projection; this polynomial model is an approximation at 150 deg.
        """
        h, w = img.shape[:2]
        fx = w / (2 * np.tan(np.radians(hfov_deg) / 2))
        fy = fx
        cx, cy = w / 2.0, h / 2.0

        # normalized pinhole (undistorted) coordinates
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        x = (xs - cx) / fx
        y = (ys - cy) / fy
        r2 = x ** 2 + y ** 2

        # forward radial + tangential distortion
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_d = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        map_x = (x_d * fx + cx).astype(np.float32)
        map_y = (y_d * fy + cy).astype(np.float32)

        return cv2.remap(
            img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    def apply(self, img: np.ndarray, pipeline: List[Tuple[str, Dict]]) -> np.ndarray:
        """Run an ordered list of (psf_name, params) steps, chaining outputs."""
        result = img
        for name, params in pipeline:
            method = getattr(self, f"apply_{name}", None)
            if method is None:
                raise ValueError(f"Unknown PSF type: {name!r}")
            result = method(result, **params)
        return result

    @staticmethod
    def _airy_kernel(radius_px: float, kernel_size: int) -> np.ndarray:
        """Build a normalized Airy-disk (jinc) convolution kernel."""
        ax = np.arange(kernel_size) - (kernel_size - 1) / 2.0
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(xx ** 2 + yy ** 2)
        # scale radial distance so the first Airy null lands at radius_px
        x = (np.pi * r / radius_px) + 1e-8
        kernel = (2 * j1(x) / x) ** 2
        kernel /= kernel.sum()
        return kernel.astype(np.float32)

    @staticmethod
    def _motion_kernel(length: int, angle: float) -> np.ndarray:
        """Build a normalized line kernel representing linear motion blur."""
        size = max(int(length), 1)
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        center = (size / 2.0, size / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
        total = kernel.sum()
        return kernel / total if total != 0 else kernel
