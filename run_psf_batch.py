#!/usr/bin/env python3
"""
run_psf_batch.py
================
Batch-apply a PSF pipeline to a folder of rendered frames, driven by a job JSON.

Written for the multi_project_pipeline `psf` stage: it degrades a copy of an
existing flythrough capture so the navigation-accuracy analysis can be repeated
under different simulated camera optics.

When the pipeline contains a ``fisheye`` step the hit-point grid is resampled
through the *same* distortion map as the image. This is required because the
accuracy analyzer treats hit-point columns 0,1 (px,py) as a pixel -> ENU lookup
table and rebuilds a regular grid from them; warping the image without warping
the hit points would silently corrupt every measurement.

Job JSON schema
---------------
{
  "input_frames":     "<dir with frame_%04d.png>",
  "output_frames":    "<dir to write to>",
  "pipeline":         [["gaussian", {"ksize": [3,3], "sigma": 0.8}], ...],
  "frames":           [10, 20, ...]        // optional, null/omitted = all
  "input_hitpoints":  "<dir with frame_%04d.npy>",   // optional
  "output_hitpoints": "<dir to write to>",           // optional
  "hitpoint_stride":  4,
  "image_width":      640,
  "image_height":     512
}

Usage:
    python run_psf_batch.py --job path/to/psf_job.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from psf import PSFSimulator

# Params that must be tuples for OpenCV but arrive from JSON as lists.
_TUPLE_PARAMS = {"ksize", "airlight"}

_FRAME_ID_RE = re.compile(r"(\d+)")


def _frame_id(path: Path):
    m = _FRAME_ID_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _coerce_params(params: dict) -> dict:
    """JSON gives lists; OpenCV needs tuples for ksize/airlight."""
    out = dict(params or {})
    for key in _TUPLE_PARAMS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def normalize_pipeline(pipeline) -> list:
    """Accept [[name, params], ...] (JSON) and return [(name, params), ...]."""
    steps = []
    for item in pipeline or []:
        if isinstance(item, dict):
            name, params = item["name"], item.get("params", {})
        else:
            name = item[0]
            params = item[1] if len(item) > 1 else {}
        steps.append((name, _coerce_params(params)))
    return steps


def fisheye_maps(width: int, height: int, k1=-0.35, k2=0.15, p1=0.0, p2=0.0,
                 k3=-0.02, hfov_deg=80.0):
    """Rebuild the exact sampling map used by PSFSimulator.apply_fisheye.

    Returns (map_x, map_y), each (height, width) float32, where destination
    pixel (u, v) samples the source image at (map_x[v, u], map_y[v, u]).
    """
    fx = width / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r2 = x ** 2 + y ** 2

    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_d = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    map_x = (x_d * fx + cx).astype(np.float32)
    map_y = (y_d * fy + cy).astype(np.float32)
    return map_x, map_y


def remap_hitpoints(hitpoints: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                    stride: int, width: int, height: int) -> np.ndarray:
    """Resample a (N,5) [px,py,E,N,U] hit-point grid through a distortion map.

    The output stays on the canonical regular lattice (px in range(0, width,
    stride), py in range(0, height, stride)), which the accuracy analyzer
    requires in order to rebuild its interpolation grid.
    """
    px = hitpoints[:, 0].astype(np.float64)
    py = hitpoints[:, 1].astype(np.float64)

    # Source lattice, exactly as the analyzer reconstructs it.
    px0, py0 = float(px.min()), float(py.min())
    col = np.round((px - px0) / stride).astype(int)
    row = np.round((py - py0) / stride).astype(int)
    n_cols, n_rows = int(col.max()) + 1, int(row.max()) + 1

    grids = []
    for c in (2, 3, 4):
        g = np.full((n_rows, n_cols), np.nan, np.float32)
        g[row, col] = hitpoints[:, c].astype(np.float32)
        grids.append(g)

    # Validity is tracked in a separate mask rather than via NaN: OpenCV's bilinear
    # filter still multiplies the out-of-bounds neighbour by a zero weight, and
    # 0 * NaN is NaN, which would discard the whole last row and column.
    mask = np.isfinite(grids[0]) & np.isfinite(grids[1]) & np.isfinite(grids[2])
    mask_f = mask.astype(np.float32)
    grids = [np.nan_to_num(g, nan=0.0) for g in grids]

    # Destination lattice (canonical, full image).
    u = np.arange(0, width, stride)
    v = np.arange(0, height, stride)
    uu, vv = np.meshgrid(u, v)

    # Where each destination node samples the source image, in source pixels...
    src_x = map_x[vv, uu].astype(np.float64)
    src_y = map_y[vv, uu].astype(np.float64)
    # ...converted to source-lattice coordinates.
    qx = ((src_x - px0) / stride).astype(np.float32)
    qy = ((src_y - py0) / stride).astype(np.float32)

    def _sample(g, border_value=0.0):
        return cv2.remap(g, qx, qy, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    sampled = [_sample(g) for g in grids]
    weight = _sample(mask_f)

    # Keep only nodes whose four contributing samples were all valid and in bounds.
    valid = weight >= 1.0 - 1e-6
    if not valid.any():
        return np.zeros((0, 5), dtype=np.float32)

    out = np.column_stack([
        uu[valid].astype(np.float32),
        vv[valid].astype(np.float32),
        sampled[0][valid],
        sampled[1][valid],
        sampled[2][valid],
    ])
    return out.astype(np.float32)


def _process_frames(job: dict, steps: list) -> int:
    in_dir = Path(job["input_frames"])
    out_dir = Path(job["output_frames"])
    if not in_dir.is_dir():
        raise SystemExit(f"input_frames not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = job.get("frames")
    wanted = set(int(i) for i in wanted) if wanted else None

    files = sorted(in_dir.glob("*.png"))
    if wanted is not None:
        files = [f for f in files if _frame_id(f) in wanted]
    if not files:
        raise SystemExit(f"no matching .png frames in {in_dir}")

    simulator = PSFSimulator()
    for i, src in enumerate(files, 1):
        dst = out_dir / src.name
        if not steps:
            shutil.copyfile(src, dst)
        else:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"could not read {src}")
            cv2.imwrite(str(dst), simulator.apply(img, steps))
        if i % 10 == 0 or i == len(files):
            print(f"  frames {i}/{len(files)}", flush=True)
    return len(files)


def _process_hitpoints(job: dict, steps: list) -> int:
    fisheye = [p for name, p in steps if name == "fisheye"]
    in_dir = job.get("input_hitpoints")
    out_dir = job.get("output_hitpoints")
    if not fisheye or not in_dir or not out_dir:
        print("  hit points: unchanged (no fisheye step)", flush=True)
        return 0
    if len(fisheye) > 1:
        raise SystemExit("multiple fisheye steps are not supported "
                         "(the hit-point remap composes only one map)")

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"input_hitpoints not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    width = int(job["image_width"])
    height = int(job["image_height"])
    stride = int(job.get("hitpoint_stride", 1) or 1)
    map_x, map_y = fisheye_maps(width, height, **fisheye[0])

    wanted = job.get("frames")
    wanted = set(int(i) for i in wanted) if wanted else None

    files = sorted(in_dir.glob("*.npy"))
    if wanted is not None:
        files = [f for f in files if _frame_id(f) in wanted]

    for i, src in enumerate(files, 1):
        hp = np.load(src)
        if hp.ndim != 2 or hp.shape[1] < 5 or len(hp) == 0:
            print(f"  WARNING: skipping malformed hit points {src.name} shape={hp.shape}")
            continue
        out = remap_hitpoints(hp, map_x, map_y, stride, width, height)
        np.save(out_dir / src.name, out)
        if i % 10 == 0 or i == len(files):
            print(f"  hit points {i}/{len(files)} (last: {len(hp)} -> {len(out)} pts)",
                  flush=True)
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch-apply a PSF pipeline to rendered frames.")
    ap.add_argument("--job", required=True, help="Path to the job .json file.")
    args = ap.parse_args(argv)

    with open(args.job, "r", encoding="utf-8") as f:
        job = json.load(f)

    steps = normalize_pipeline(job.get("pipeline"))
    print(f"PSF pipeline: {[n for n, _ in steps] or ['<none - straight copy>']}", flush=True)

    n_frames = _process_frames(job, steps)
    n_hits = _process_hitpoints(job, steps)
    print(f"PSF done: {n_frames} frames, {n_hits} hit-point files rewritten.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
