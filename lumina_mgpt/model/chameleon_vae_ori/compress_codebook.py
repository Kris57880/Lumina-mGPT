"""Utilities for compressing a VQ-GAN codebook via clustering.

This script clusters the codebook embeddings extracted from a checkpoint and
exports remap arrays that can be consumed by ``VectorQuantizer2``'s ``remap``
mechanism at inference time.

Usage example (compression to multiple target sizes)::

    python compress_codebook.py \
        --ckpt /path/to/vq_model.ckpt \
        --output-dir ./compressed_codebooks \
        --k-values 1024 2048 4096 \
        --method kmeans

The generated ``remap_k*.npy`` files can be passed to ``VectorQuantizer2`` via
its ``remap`` argument to restrict the latent codes that appear at inference
without modifying the underlying model weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

LOG = logging.getLogger("compress_codebook")


# ---------------------------------------------------------------------------
# Embedding loading utilities
# ---------------------------------------------------------------------------

def _resolve_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return the underlying state dict from a checkpoint dictionary."""

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


def _infer_embedding_key(state_dict: Dict[str, torch.Tensor]) -> str:
    candidates = [k for k, v in state_dict.items() if k.endswith("embedding.weight") and v.ndim == 2]
    if not candidates:
        raise ValueError("No parameter ending with 'embedding.weight' found in checkpoint."
                         " Please supply --embedding-key explicitly.")
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Multiple embedding weights detected. Please select one via --embedding-key. Candidates: "
        + ", ".join(candidates)
    )


def load_codebook_embeddings(
    ckpt_path: Path,
    embedding_key: Optional[str] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load the VQ codebook embeddings from a checkpoint."""

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint file does not contain a state dictionary.")

    state_dict = _resolve_state_dict(checkpoint)

    if embedding_key is None:
        embedding_key = _infer_embedding_key(state_dict)
        LOG.info("Auto-selected embedding parameter: %s", embedding_key)

    if embedding_key not in state_dict:
        raise KeyError(f"Embedding key '{embedding_key}' not found in checkpoint.")

    embeddings = state_dict[embedding_key].detach().float()  # (num_codes, dim)
    LOG.info("Loaded embeddings with shape %s", tuple(embeddings.shape))

    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        LOG.info("Applied L2 normalization to embeddings.")

    return embeddings


# ---------------------------------------------------------------------------
# Clustering backends
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    centers: np.ndarray  # (k, dim)
    labels: np.ndarray  # (num_codes,)


class Clusterer:
    """Abstract clustering strategy."""

    name: str

    def fit(self, embeddings: np.ndarray, k: int) -> ClusterResult:
        raise NotImplementedError


class KMeansClusterer(Clusterer):
    """KMeans clustering using scikit-learn's MiniBatchKMeans."""

    def __init__(self, batch_size: int = 4096, max_iter: int = 300, tol: float = 1e-4, random_state: Optional[int] = 0, **_unused):
        self.name = "kmeans"
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, embeddings: np.ndarray, k: int) -> ClusterResult:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "scikit-learn is required for the KMeans clustering backend. Please install it via 'pip install scikit-learn'."
            ) from exc

        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(self.batch_size, embeddings.shape[0]),
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_
        return ClusterResult(centers=centers.astype(np.float32), labels=labels.astype(np.int64))


class DKMClusterer(Clusterer):
    """Differentiable k-means based on soft assignments."""

    def __init__(
        self,
        tau: float = 0.1,
        iters: int = 200,
        init: str = "kmeans++",
        eps: float = 1e-8,
        requires_grad: bool = False,
        temp_anneal: Optional[Tuple[float, float, int]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **_unused,
    ):
        self.name = "dkm"
        self.tau = tau
        self.iters = iters
        self.init = init
        self.eps = eps
        self.requires_grad = requires_grad
        self.temp_anneal = temp_anneal
        self.random_state = random_state
        self.verbose = verbose

    def _kmeans_pp_init(self, embeddings: torch.Tensor, k: int, seed: Optional[int]) -> torch.Tensor:
        K, dim = embeddings.shape
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        centers = torch.empty((k, dim), device=embeddings.device, dtype=embeddings.dtype)
        idx = random.randrange(0, K)
        centers[0] = embeddings[idx]
        closest_dist_sq = torch.sum((embeddings - centers[0]) ** 2, dim=1)
        for c in range(1, k):
            probs = closest_dist_sq / torch.sum(closest_dist_sq)
            idx = torch.multinomial(probs, 1).item()
            centers[c] = embeddings[idx]
            dist_sq = torch.sum((embeddings - centers[c]) ** 2, dim=1)
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
        return centers

    def _dkm(self, embeddings: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        device = embeddings.device
        K, dim = embeddings.shape

        if self.init == "kmeans++":
            centers = self._kmeans_pp_init(embeddings, k, self.random_state)
        elif self.init == "random":
            idx = torch.randperm(K, device=device)[:k]
            centers = embeddings[idx].clone()
        else:
            raise ValueError("init must be 'kmeans++' or 'random'")

        centers = centers.clone().detach()
        centers.requires_grad_(self.requires_grad)

        if self.temp_anneal is not None:
            tau_start, tau_end, anneal_iters = self.temp_anneal
        else:
            tau_start = tau_end = self.tau
            anneal_iters = 0

        for t in range(self.iters):
            if t < anneal_iters:
                cur_tau = tau_start + (tau_end - tau_start) * (t / max(1, anneal_iters - 1))
            else:
                cur_tau = tau_end

            embeddings_sq = torch.sum(embeddings * embeddings, dim=1, keepdim=True)
            centers_sq = torch.sum(centers * centers, dim=1, keepdim=True).T
            distances = embeddings_sq + centers_sq - 2.0 * (embeddings @ centers.T)

            logits = -distances / (cur_tau + 1e-12)
            assignments = torch.softmax(logits, dim=1)

            denom = assignments.sum(dim=0, keepdim=True).T + self.eps
            new_centers = (assignments.T @ embeddings) / denom

            if not self.requires_grad:
                centers = new_centers.detach()
            else:
                centers = new_centers

            if self.verbose and (t % max(1, self.iters // 10) == 0):
                total_assignments = assignments.sum()
                if total_assignments > 0:
                    mean_sq_dist = (assignments * distances).sum() / total_assignments
                    mean_dist = float(torch.sqrt(mean_sq_dist))
                    max_dist = float(torch.sqrt(distances.max()))
                else:
                    mean_dist = float("nan")
                    max_dist = float("nan")
                LOG.info(
                    "DKM iter %d/%d, tau=%.4f, mean_distance=%.6f, max_distance=%.6f",
                    t,
                    self.iters,
                    cur_tau,
                    mean_dist,
                    max_dist,
                )

        embeddings_sq = torch.sum(embeddings * embeddings, dim=1, keepdim=True)
        centers_sq = torch.sum(centers * centers, dim=1, keepdim=True).T
        distances = embeddings_sq + centers_sq - 2.0 * (embeddings @ centers.T)
        logits = -distances / (cur_tau + 1e-12)
        assignments = torch.softmax(logits, dim=1)

        return centers, assignments

    def fit(self, embeddings: np.ndarray, k: int) -> ClusterResult:
        emb_tensor = torch.from_numpy(embeddings).to(torch.float32)
        centers, assignments = self._dkm(emb_tensor, k)
        hard_labels = torch.argmax(assignments, dim=1).cpu().numpy().astype(np.int64)
        return ClusterResult(centers=centers.cpu().numpy().astype(np.float32), labels=hard_labels)


def build_clusterer(method: str, **kwargs) -> Clusterer:
    method = method.lower()
    if method == "kmeans":
        return KMeansClusterer(**kwargs)
    if method == "dkm":
        return DKMClusterer(**kwargs)
    raise ValueError(f"Unsupported clustering method '{method}'." )


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def choose_representatives(
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
) -> np.ndarray:
    """Select a representative code index for each cluster center."""

    k = cluster_result.centers.shape[0]
    representatives: List[int] = []

    empty_clusters = 0

    for cluster_id in range(k):
        members = np.where(cluster_result.labels == cluster_id)[0]
        if members.size == 0:
            # LOG.warning("Cluster %d is empty; falling back to nearest global vector.", cluster_id)
            # Find closest vector in entire set to cluster center
            dists = np.linalg.norm(embeddings - cluster_result.centers[cluster_id], axis=1)
            representatives.append(int(np.argmin(dists)))
            empty_clusters += 1
            continue

        member_vectors = embeddings[members]
        dists = np.linalg.norm(member_vectors - cluster_result.centers[cluster_id], axis=1)
        best_idx = members[int(np.argmin(dists))]
        representatives.append(int(best_idx))

    cluster_result.empty_clusters = empty_clusters  # type: ignore[attr-defined]

    return np.array(representatives, dtype=np.int64)


def compute_distortion(
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
) -> Dict[str, float]:
    """Compute reconstruction metrics for the clustering."""

    centers = cluster_result.centers
    labels = cluster_result.labels
    dists = np.linalg.norm(embeddings - centers[labels], axis=1)
    return {
        "mean_distance": float(np.mean(dists)),
        "max_distance": float(np.max(dists)),
        "std_distance": float(np.std(dists)),
        "median_distance": float(np.median(dists)),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_numpy_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    LOG.info("Saved %s (shape=%s)", path, array.shape)


def save_summary(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOG.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress a VQ codebook by clustering and export remap arrays.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the checkpoint containing the codebook embeddings.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where remap files will be written.")
    parser.add_argument("--k-values", type=int, nargs="+", required=True, help="Target codebook sizes to generate.")
    parser.add_argument("--embedding-key", type=str, default=None, help="State dict key for the embedding weight.")
    parser.add_argument("--method", type=str, default="kmeans", help="Clustering backend to use (default: kmeans).")
    parser.add_argument("--normalize", action="store_true", help="Apply L2 normalization before clustering (default: False).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for clustering backends that support it.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size for clustering backends.")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum iterations for clustering backends.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for clustering convergence.")
    parser.add_argument("--tau", type=float, default=0.1, help="Softmax temperature for DKM (smaller -> harder assignments).")
    parser.add_argument("--dkm-iters", type=int, default=200, help="Number of iterations for DKM.")
    parser.add_argument("--dkm-init", type=str, default="kmeans++", choices=["kmeans++", "random"], help="Initialization strategy for DKM.")
    parser.add_argument("--dkm-anneal", type=str, default=None, help="Temperature annealing schedule for DKM as tau_start,tau_end,steps.")
    parser.add_argument("--dkm-verbose", action="store_true", help="Enable verbose logging for DKM iterations.")
    parser.add_argument("--write-mapping", action="store_true", help="Also save the full cluster assignment mapping (per-k).")
    parser.add_argument("--summary", action="store_true", help="Write a JSON summary with distortion statistics per k.")
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Save a histogram (bar chart) of cluster membership frequencies for each k.",
    )
    return parser.parse_args(argv)


def plot_cluster_histogram(labels: np.ndarray, k: int, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to export histograms. Install it via 'pip install matplotlib'."
        ) from exc

    counts = np.bincount(labels, minlength=k)
    x = np.arange(k)

    width = max(6.0, min(24.0, k / 64))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(x, counts, width=1.0)
    ax.set_title(f"Cluster membership distribution (k={k})")
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Assignment count")
    ax.set_xlim(-0.5, k - 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved histogram to %s", output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    # --- Cluster 可視化 ---
    import matplotlib.pyplot as plt
    def visualize_token_clusters(mapping, k, save_path):
        num_tokens = mapping.shape[0]
        side = int(np.ceil(np.sqrt(num_tokens)))
        arr = np.full((side * side,), -1, dtype=np.int32)
        arr[:num_tokens] = mapping
        arr = arr.reshape(side, side)
        unique_clusters = np.unique(mapping)
        palette = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_clusters)))[:, :3]
        color_map = {cid: palette[i] for i, cid in enumerate(unique_clusters)}
        rgb_img = np.zeros((side, side, 3), dtype=np.float32)
        for cid in unique_clusters:
            rgb_img[arr == cid] = color_map[cid]
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_img)
        plt.title(f"Token cluster visualization (k={k})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    normalize = args.normalize

    embeddings = load_codebook_embeddings(
        ckpt_path=args.ckpt,
        embedding_key=args.embedding_key,
        normalize=normalize,
    )

    embeddings_np = embeddings.cpu().numpy()

    anneal = None
    if args.dkm_anneal:
        try:
            parts = [float(x) for x in args.dkm_anneal.split(",")]
            if len(parts) != 3:
                raise ValueError
            anneal = (parts[0], parts[1], int(parts[2]))
        except ValueError as exc:
            raise ValueError("--dkm-anneal must be formatted as tau_start,tau_end,steps") from exc

    clusterer = build_clusterer(
        args.method,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        tau=args.tau,
        iters=args.dkm_iters,
        init=args.dkm_init,
        temp_anneal=anneal,
        verbose=args.dkm_verbose,
    )

    summary: Dict[str, Dict[str, float]] = {}

    for k in args.k_values:
        if k <= 0:
            LOG.warning("Skipping invalid k=%d (must be positive).", k)
            continue
        if k > embeddings_np.shape[0]:
            LOG.warning(
                "Requested k=%d exceeds the number of available codes (%d). Reducing to %d.",
                k,
                embeddings_np.shape[0],
                embeddings_np.shape[0],
            )
            k = embeddings_np.shape[0]

        LOG.info("Running %s clustering with k=%d", clusterer.name, k)
        cluster_result = clusterer.fit(embeddings_np, k)
        counts = np.bincount(cluster_result.labels, minlength=k)
        LOG.info("Top 10 cluster sizes for k=%d: %s", k, np.sort(counts)[::-1][:10])
        representatives = choose_representatives(embeddings_np, cluster_result)

        remap_path = args.output_dir / f"remap_{clusterer.name}_k{k}.npy"
        save_numpy_array(remap_path, representatives)

        if args.write_mapping:
            mapping_path = args.output_dir / f"mapping_{clusterer.name}_k{k}.npy"
            save_numpy_array(mapping_path, cluster_result.labels.astype(np.int64))

            # 直接視覺化 mapping，不用 argument 指定路徑
            vis_path = mapping_path.parent / f"vis_token_clusters_k{k}.png"
            visualize_token_clusters(cluster_result.labels, k, vis_path)

        if args.histogram:
            hist_path = args.output_dir / f"histogram_{clusterer.name}_k{k}.png"
            plot_cluster_histogram(cluster_result.labels, k, hist_path)

        distortion = compute_distortion(embeddings_np, cluster_result)
        empty_count = getattr(cluster_result, "empty_clusters", None)
        if empty_count is not None:
            distortion["empty_clusters"] = int(empty_count)
        summary[f"k={k}"] = distortion
        LOG.info(
            "Distortion for k=%d -> mean=%.6f std=%.6f max=%.6f median=%.6f",
            k,
            distortion["mean_distance"],
            distortion["std_distance"],
            distortion["max_distance"],
            distortion["median_distance"],
        )

    if args.summary and summary:
        summary_path = args.output_dir / f"summary_{clusterer.name}.json"
        save_summary(summary_path, summary)


if __name__ == "__main__":
    main()
