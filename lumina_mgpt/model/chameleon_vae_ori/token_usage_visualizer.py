#!/usr/bin/env python3
"""Visualize VQ token usage frequencies over a folder of images.

This minimal utility encodes every image in a directory with the VQModel,
(optionally) applies a remapped codebook, decodes the latents once for sanity,
and records how frequently each token index appears. The final frequencies are
rendered as a square heatmap using a single-hue colormap where lighter colors
represent higher usage.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

try:  # Support both package and standalone execution
    from .vqgan import VQModel
except ImportError:  # pragma: no cover - fallback for script usage
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from lumina_mgpt.model.chameleon_vae_ori.vqgan import VQModel

LOG = logging.getLogger("token_usage")


def _load_model(config_path: Path, ckpt_path: Path, remap_path: Optional[Path], device: Optional[str]) -> VQModel:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    params = dict(config["model"]["params"])
    params.pop("lossconfig", None)
    params["ckpt_path"] = str(ckpt_path)
    if remap_path is not None:
        params["remap"] = str(remap_path)
        params.setdefault("sane_index_shape", True)

    model = VQModel(**params)
    model.eval()

    if device is None:
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1:
            raise RuntimeError("Model parameters reside on multiple devices; please specify --device explicitly.")
        device = devices.pop()
    else:
        model.to(device)
        device = torch.device(device)

    LOG.info("Loaded VQModel on %s (remap=%s)", device, remap_path if remap_path else "none")
    return model


def _whiten_transparency(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img

    rgba = np.array(img.convert("RGBA"))
    if not (rgba[:, :, 3] < 255).any():
        return img.convert("RGB")

    alpha = rgba[:, :, 3] / 255.0
    rgb = (1 - alpha[..., None]) * 255 + alpha[..., None] * rgba[:, :, :3]
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _image_to_tensor(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = _whiten_transparency(img)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _encode_tokens(model: VQModel, image_path: Path) -> tuple[torch.Tensor, int, int]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    img = Image.open(image_path)
    img.resize((1024, 1024), Image.BICUBIC)  
    tensor = _image_to_tensor(img, device=device, dtype=dtype)

    with torch.no_grad():
        quantized, _, (_, _, indices) = model.encode(tensor)

    b, _, h, w = quantized.shape
    if b != 1:
        raise ValueError("Batch size greater than 1 is not supported for this script.")

    indices = indices.view(-1).to(device=device, dtype=torch.long)

    # Optional decode to ensure pass-through.
    with torch.no_grad():
        e_dim = model.quantize.e_dim
        recon = model.quantize.get_codebook_entry(indices, (1, h, w, e_dim))
        _ = model.decode(recon)  # result unused; ensures decode succeeds

    return indices, h, w


def _iter_image_files(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _visualize_frequencies(freq: np.ndarray, output_path: Path) -> None:
    total_codes = freq.shape[0]
    side = int(math.ceil(math.sqrt(total_codes)))

    canvas = np.zeros(side * side, dtype=np.float32)
    canvas[:total_codes] = freq
    canvas = canvas.reshape(side, side)

    if canvas.max() > 0:
        norm_canvas = canvas / canvas.max()
    else:
        norm_canvas = canvas

    plt.figure(figsize=(8, 8))
    im = plt.imshow(norm_canvas, interpolation="nearest")
    plt.title("Token usage frequency")
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized frequency", rotation=-90, va="bottom")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    LOG.info("Saved visualization to %s", output_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VQ token usage across a folder of images.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder containing input images.")
    parser.add_argument("--config", type=Path, default=Path("./ckpts/chameleon/tokenizer/vqgan.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("./ckpts/chameleon/tokenizer/vqgan.ckpt"))
    parser.add_argument("--remap", type=Path, default=None, help="Optional remap numpy file for compressed codebook.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--output", type=Path, default=Path("./token_usage.png"), help="Output image for visualization.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    model = _load_model(args.config, args.ckpt, args.remap, args.device)

    if args.remap is not None:
        remap_array = np.load(args.remap)
        total_tokens = int(remap_array.shape[0])
    else:
        total_tokens = int(model.quantize.n_e)

    frequencies = np.zeros(total_tokens, dtype=np.int64)
    processed = 0

    for image_path in _iter_image_files(args.image_dir):
        indices, _, _ = _encode_tokens(model, image_path)
        freq_np = indices.detach().cpu().numpy().astype(np.int64)

        if freq_np.max(initial=0) >= total_tokens:
            LOG.warning("Encountered token index %d beyond expected range %d", freq_np.max(), total_tokens)
            continue

        np.add.at(frequencies, freq_np, 1)
        processed += 1
        LOG.info("Processed %s (total=%d)", image_path.name, processed)

    if processed == 0:
        LOG.warning("No images processed; skipping visualization.")
        return

    _visualize_frequencies(frequencies.astype(np.float32), args.output)


if __name__ == "__main__":
    main()
