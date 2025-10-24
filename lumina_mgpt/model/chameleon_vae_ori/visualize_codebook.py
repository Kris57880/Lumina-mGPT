"""Visualization utilities for VQ codebook remap / clustering results.

This script projects codebook embeddings to 2D (via PCA or t-SNE) and colors the
points by their cluster assignment, enabling an at-a-glance inspection of the
compressed codebook mapping.

Example::

    python visualize_codebook.py \
        --ckpt /path/to/vqgan.ckpt \
        --remap ./codebook_remap/remap_kmeans_k1024.npy \
        --mapping ./codebook_remap/mapping_kmeans_k1024.npy \
        --output ./codebook_remap/vis_k1024.png \
        --dim-red tsne \
        --sample-size 10000

If ``--mapping`` is omitted, cluster membership can be reconstructed from the
remap array using ``--infer-membership`` (each point is assigned to the nearest
representative code specified in the remap array).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from compress_codebook import load_codebook_embeddings

LOG = logging.getLogger("visualize_codebook")


def load_remap(path: Path) -> np.ndarray:
    remap = np.load(path)
    if remap.ndim != 1:
        raise ValueError("Remap array is expected to be 1-D (new_index -> original_index).")
    return remap.astype(np.int64)


def load_mapping(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    mapping = np.load(path)
    if mapping.ndim != 1:
        raise ValueError("Mapping array should be 1-D (original_index -> cluster_id).")
    return mapping.astype(np.int64)


def infer_membership_from_remap(
    embeddings: np.ndarray,
    remap: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Assign each embedding to the nearest representative defined by remap."""

    reps = embeddings[remap]  # (k, dim)
    num_vectors = embeddings.shape[0]
    k = reps.shape[0]
    memberships = np.empty(num_vectors, dtype=np.int64)

    # Process in batches to limit memory usage
    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        chunk = embeddings[start:end]
        # (chunk_size, dim) @ (dim, k) -> (chunk_size, k)
        # Use cosine distance if embeddings are normalized (approx by dot product)
        # fallback to Euclidean otherwise
        # We'll compute Euclidean: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        reps_sq = np.sum(reps * reps, axis=1)
        distances = chunk_sq + reps_sq - 2.0 * np.dot(chunk, reps.T)
        memberships[start:end] = np.argmin(distances, axis=1)

    LOG.info("Inferred memberships for %d embeddings across %d clusters", num_vectors, k)
    return memberships


def reduce_dimension(
    embeddings: np.ndarray,
    method: str,
    sample_size: Optional[int],
    random_state: int,
    output_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (subset_indices, transformed_points) with ``output_dims`` dimensions."""

    rng = np.random.default_rng(random_state)
    num_points = embeddings.shape[0]

    if sample_size is not None and sample_size < num_points:
        indices = rng.choice(num_points, size=sample_size, replace=False)
        embeddings_subset = embeddings[indices]
        subset_indices = indices
    else:
        embeddings_subset = embeddings
        subset_indices = np.arange(num_points)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=output_dims, random_state=random_state)
        reduced = reducer.fit_transform(embeddings_subset)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=output_dims,
            init="pca" if output_dims <= 3 else "random",
            learning_rate="auto",
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings_subset)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method '{method}'.")

    return subset_indices, reduced


def build_color_palette(num_colors: int, seed: int) -> ListedColormap:
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.0, 1.0, size=(num_colors, 3))
    return ListedColormap(colors)


def plot_clusters(
    reduced_points: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
    marker_size: float = 4.0,
) -> None:
    unique_labels = np.unique(labels)
    cmap = build_color_palette(len(unique_labels), seed)

    is_3d = reduced_points.shape[1] == 3
    fig = plt.figure(figsize=(9, 7))
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced_points[:, 0],
            reduced_points[:, 1],
            reduced_points[:, 2],
            c=labels,
            cmap=cmap,
            s=marker_size,
            linewidths=0,
            alpha=0.8,
        )
        ax.set_zlabel("Component 3")
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            reduced_points[:, 0],
            reduced_points[:, 1],
            c=labels,
            cmap=cmap,
            s=marker_size,
            linewidths=0,
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    legend_cap = min(len(unique_labels), 20)
    if legend_cap > 0:
        legend_labels = [str(lab) for lab in unique_labels[:legend_cap]]
        legend_handles = scatter.legend_elements(num=legend_cap)[0][:legend_cap]
        ax.legend(legend_handles, legend_labels, title="Cluster IDs", loc="best", fontsize="small")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved visualization to %s", output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustered VQ codebook embeddings.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the VQ checkpoint with embeddings.")
    parser.add_argument("--remap", type=Path, required=True, help="Path to remap numpy file (new_index -> original_index).")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional numpy file mapping original indices to cluster IDs.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the visualization image.")
    parser.add_argument("--embedding-key", type=str, default=None, help="Optional embedding key in the checkpoint.")
    parser.add_argument("--dim-red", type=str, choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method.")
    parser.add_argument("--sample-size", type=int, default=None, help="Random subset size for plotting (default: all points).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and color palette.")
    parser.add_argument(
        "--plot-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of dimensions to visualize (2 for 2D scatter, 3 for 3D scatter).",
    )
    parser.add_argument(
        "--infer-membership",
        action="store_true",
        help="Derive cluster membership from remap when mapping file is not provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for membership inference (only if --infer-membership).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    embeddings = load_codebook_embeddings(
        ckpt_path=args.ckpt,
        embedding_key=args.embedding_key,
        normalize=True,  # For visualization we default to normalized vectors
    ).cpu().numpy()

    remap = load_remap(args.remap)
    mapping = load_mapping(args.mapping)

    if mapping is not None:
        LOG.info("Loaded mapping with %d assignments.", mapping.size)
        memberships = mapping
    else:
        if not args.infer_membership:
            raise ValueError(
                "Mapping file not provided. Either supply --mapping or enable --infer-membership to derive assignments."
            )
        memberships = infer_membership_from_remap(
            embeddings=embeddings,
            remap=remap,
            batch_size=args.batch_size,
        )

    indices_subset, reduced = reduce_dimension(
        embeddings,
        method=args.dim_red,
        sample_size=args.sample_size,
        random_state=args.seed,
        output_dims=args.plot_dims,
    )

    subset_labels = memberships[indices_subset]

    title = f"Codebook visualization ({args.dim_red.upper()}, {args.plot_dims}D), k={remap.shape[0]}"
    plot_clusters(
        reduced_points=reduced,
        labels=subset_labels,
        output_path=args.output,
        title=title,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
