"""Visualize a remap array produced by `compress_codebook.py` as colored square grids.

Given a 1-D NumPy array `remap.npy` where each entry maps a compressed codebook index
(`new_idx`) to an original code index, this script renders a side-by-side comparison using
high-contrast colored tiles:

- Left subplot: tiles labeled by the compressed index order (0..k-1).
- Right subplot: tiles colored by the corresponding original index stored in the remap.

Both tiles are arranged in a square grid whose size is determined by the number of entries.
Unused cells (when k is not a perfect square) are left blank. This provides a quick visual
way to inspect how codes are regrouped after compression.

Usage::

    python visualize_remap_grid.py \
        --remap ./codebook_remap/remap_dkm_k4096.npy \
        --output ./codebook_remap/vis/remap_grid_k4096.png \
        --palette-seed 42

The script requires ``matplotlib`` and ``numpy``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_remap(path: Path) -> np.ndarray:
    remap = np.load(path)
    if remap.ndim != 1:
        raise ValueError("Remap array must be 1-D (new_index -> original_index).")
    return remap.astype(np.int64)


def _load_mapping(path: Path) -> np.ndarray:
    mapping = np.load(path)
    if mapping.ndim != 1:
        raise ValueError("Mapping array must be 1-D (original_index -> cluster_id).")
    return mapping.astype(np.int64)


def _square_grid(values: np.ndarray, fill_value: float = np.nan) -> tuple[np.ndarray, int, int]:
    total = int(values.shape[0])
    side = max(1, math.ceil(math.sqrt(total)))
    dtype = float if np.isnan(fill_value) else values.dtype
    grid = np.full((side * side,), fill_value, dtype=dtype)
    grid[:total] = values.astype(dtype)
    grid = grid.reshape(side, side)
    return grid, side, side


def _build_palette(num_colors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, num_colors, endpoint=False)
    rng.shuffle(hues)
    colors = []
    for hue in hues:
        r, g, b, _ = plt.cm.hsv(hue)
        colors.append((r, g, b))
    return np.array(colors)


def _plot_grid(ax: plt.Axes, grid: np.ndarray, palette: np.ndarray, title: str) -> None:
    mask = np.isnan(grid)
    shown = grid.copy()
    shown[mask] = 0

    norm = plt.Normalize(vmin=0, vmax=max(1, palette.shape[0] - 1))
    cmap = plt.cm.colors.ListedColormap(palette)

    ax.imshow(shown, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis("off")

    if mask.any():
        ax.imshow(mask, cmap="gray", alpha=0.1)


def visualize_remap(
    remap_path: Path,
    output_path: Path,
    palette_seed: int,
    mapping_path: Optional[Path] = None,
) -> None:
    remap = _load_remap(remap_path)
    total_codes = remap.shape[0]

    mapping = _load_mapping(mapping_path) if mapping_path is not None else None

    new_indices = np.arange(total_codes)
    new_grid, _, _ = _square_grid(new_indices)
    original_grid, _, _ = _square_grid(remap.astype(float))

    num_clusters = int(remap.shape[0])
    if mapping is not None:
        num_clusters = max(num_clusters, int(mapping.max() + 1))

    num_colors = max(num_clusters, int(remap.max() + 1))
    palette = _build_palette(num_colors, palette_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mapping is not None:
        mapping_grid, _, _ = _square_grid(mapping.astype(float))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        _plot_grid(axes[0], mapping_grid, palette, "Original codes → cluster id")
        _plot_grid(axes[1], new_grid.astype(float), palette, "Compressed indices (cluster id)")
        _plot_grid(axes[2], original_grid, palette, "Cluster representatives (remap)")

        # Print quick summary of cluster sizes
        cluster_counts = np.bincount(mapping, minlength=num_clusters)
        top_sizes = np.sort(cluster_counts)[::-1][:10]
        print("Top cluster sizes:", top_sizes)
        if (cluster_counts == 0).any():
            print("Empty clusters:", int((cluster_counts == 0).sum()))

        fig.suptitle(
            f"Remap visualization with mapping (codes={mapping.shape[0]}, clusters={num_clusters})",
            fontsize=14,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        _plot_grid(axes[0], new_grid.astype(float), palette, "Compressed indices")
        _plot_grid(axes[1], original_grid, palette, "Original indices (remap)")
        fig.suptitle(f"Remap visualization (k={total_codes})", fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a remap array as colored grids.")
    parser.add_argument("--remap", type=Path, required=True, help="Path to remap numpy file (new_index -> original_index).")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional mapping numpy file (original_index -> cluster id).")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the visualization image.")
    parser.add_argument("--palette-seed", type=int, default=0, help="Seed for randomized high-contrast palette.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    visualize_remap(
        remap_path=args.remap,
        output_path=args.output,
        palette_seed=args.palette_seed,
        mapping_path=args.mapping,
    )


if __name__ == "__main__":
    main()
