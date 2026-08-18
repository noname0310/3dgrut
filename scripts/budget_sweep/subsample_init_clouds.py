#!/usr/bin/env python3
"""Subsample COLMAP points3D.bin files to exact point counts for budget-limited
MCMC training (initialization.method=fused_point_cloud).

For each scene, budgets below the original COLMAP point count need an init
cloud with exactly `budget` points (MCMC only *adds* gaussians up to
strategy.add.max_n_gaussians, it never removes them).

Subsamples are nested (smaller budgets are subsets of larger ones) and
deterministic (seed matches seed_initialization=42 in the training configs).

Outputs: data/init_clouds/<scene>_<N>.ply  (binary little-endian, x y z red green blue)
"""

import os
import struct

import numpy as np

SEED = 42
OUT_DIR = "data/init_clouds"

# scene -> (points3D.bin path, original count, budgets below the count)
SCENES = {
    "bicycle": ("data/mipnerf360/bicycle/sparse/0/points3D.bin", 54275),
    "bonsai": ("data/mipnerf360/bonsai/sparse/0/points3D.bin", 206613),
    "garden": ("data/mipnerf360/garden/sparse/0/points3D.bin", 138766),
    "drjohnson": ("data/db/drjohnson/sparse/0/points3D.bin", 80861),
    "playroom": ("data/db/playroom/sparse/0/points3D.bin", 37005),
    "train": ("data/tandt/train/sparse/0/points3D.bin", 182686),
    "truck": ("data/tandt/truck/sparse/0/points3D.bin", 136029),
}

BUDGETS = [500000, 250000, 125000, 62500, 31250, 15625, 7812, 3906, 1953, 976]


def read_points3d_bin(path):
    """Parse COLMAP points3D.bin (same layout as threedgrut/model/model.py)."""
    with open(path, "rb") as f:
        (n_pts,) = struct.unpack("<Q", f.read(8))
        pts = np.zeros((n_pts, 3), dtype=np.float32)
        rgb = np.zeros((n_pts, 3), dtype=np.uint8)
        for i in range(n_pts):
            rec = f.read(43)
            _, x, y, z, r, g, b, _ = struct.unpack("<QdddBBBd", rec)
            pts[i] = (x, y, z)
            rgb[i] = (r, g, b)
            (t_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * t_len)
    return pts, rgb


def write_ply(path, pts, rgb):
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fffBBB", *pts[i], *rgb[i]))


def main():
    for scene, (bin_path, n_orig) in SCENES.items():
        budgets = [b for b in BUDGETS if b < n_orig]
        if not budgets:
            print(f"{scene}: {n_orig} points, no subsampling needed")
            continue
        pts, rgb = read_points3d_bin(bin_path)
        assert len(pts) == n_orig, f"{scene}: expected {n_orig} points, got {len(pts)}"
        # One permutation per scene -> nested subsamples (first N entries).
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_orig)
        for n in budgets:
            out = os.path.join(OUT_DIR, f"{scene}_{n}.ply")
            idx = np.sort(perm[:n])
            write_ply(out, pts[idx], rgb[idx])
            print(f"{scene}: wrote {out} ({n} points)")


if __name__ == "__main__":
    main()
