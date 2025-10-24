"""Encode and decode an image using a compressed (remapped) VQ codebook.

This utility loads a ``VQModel`` with a remapped codebook, tokenizes an input
image, and reconstructs it purely from the compressed code indices. The tool is
meant for inference-time verification of codebook compression artifacts.

Example::

Examples::

    # Use compressed codebook remap
    python remap_inference_tool.py \
        --config lumina_mgpt/configs/chameleon_vae.yaml \
        --ckpt lumina_mgpt/ckpts/chameleon/tokenizer/vqgan.ckpt \
        --remap ./codebook_remap/remap_kmeans_k1024.npy \
        --image ./assets/example.png \
        --output-image ./outputs/recon_k1024.png

    # Use the original (uncompressed) codebook
    python remap_inference_tool.py \
        --config lumina_mgpt/configs/chameleon_vae.yaml \
        --ckpt lumina_mgpt/ckpts/chameleon/tokenizer/vqgan.ckpt \
        --image ./assets/example.png \
        --output-image ./outputs/recon_full.png
"""

from __future__ import annotations

import argparse
import logging
import colorsys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import yaml
import matplotlib.pyplot as plt

try:  # Support both package and standalone execution
    from .vqgan import VQModel
except ImportError:  # pragma: no cover - fallback for script usage
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from lumina_mgpt.model.chameleon_vae_ori.vqgan import VQModel

LOG = logging.getLogger("remap_inference")


def _load_model(
    config_path: Path,
    ckpt_path: Path,
    device: Optional[str | torch.device],
) -> VQModel:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    params = dict(config["model"]["params"])
    params.pop("lossconfig", None)
    params["ckpt_path"] = str(ckpt_path)
    

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


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(-1.0, 1.0).cpu()
    tensor = (tensor + 1.0) / 2.0
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _ensure_output_directory(target: Path) -> Path:
    if target.suffix:
        directory = target.parent
    else:
        directory = target
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _determine_remap_label(remap_path: Optional[Path]) -> Tuple[str, Optional[int]]:
    if remap_path is None:
        return "full", None

    remap_array = np.load(remap_path)
    if remap_array.ndim != 1:
        raise ValueError("Remap array must be 1-D to determine codebook size.")
    k = int(remap_array.shape[0])
    return f"k{k}", k


def _generate_contrast_palette(num_colors: int, seed: int) -> np.ndarray:
    if num_colors <= 0:
        raise ValueError("num_colors must be positive")

    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, num_colors, endpoint=False)
    rng.shuffle(hues)

    colors = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(float(hue), 0.85, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return np.array(colors, dtype=np.uint8)


def _save_latent_visualization(
    tokens: torch.Tensor,
    height: int,
    width: int,
    output_path: Path,
    palette_seed: int,
) -> Image.Image:
    token_cpu = tokens.detach().cpu().view(height, width).numpy()
    unique_tokens = np.unique(token_cpu)

    palette = _generate_contrast_palette(len(unique_tokens), palette_seed)
    # Map tokens to palette indices via searchsorted
    token_indices = np.searchsorted(unique_tokens, token_cpu.ravel())
    color_image = palette[token_indices].reshape(height, width, 3)

    image = Image.fromarray(color_image.astype(np.uint8), mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # image.save(output_path)
    # LOG.info(
    #     "Saved latent token visualization to %s (unique tokens=%d)",
    #     output_path,
    #     unique_tokens.size,
    # )
    return image


def encode_image(model: VQModel, image_path: Path) -> tuple[torch.Tensor, int, int]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    img = Image.open(image_path)
    tensor = _image_to_tensor(img, device=device, dtype=dtype)

    with torch.no_grad():
        quantized, _, (_, _, indices) = model.encode(tensor)

    b, _, h, w = quantized.shape
    if b != 1:
        raise ValueError("Batch size greater than 1 is not supported for this demo.")

    indices = indices.view(-1).to(device=device, dtype=torch.long)
    LOG.info("Encoded image to %d tokens (latent grid %dx%d)", indices.numel(), h, w)
    return indices, h, w


def decode_tokens(model: VQModel, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    device = next(model.parameters()).device
    e_dim = model.quantize.e_dim

    with torch.no_grad():
        quant = model.quantize.get_codebook_entry(tokens.to(device=device, dtype=torch.long), (1, h, w, e_dim))
        recon = model.decode(quant)[0]
    return recon.cpu()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode and decode an image using a compressed VQ codebook.")
    parser.add_argument(
        "--config",
        type=Path,
        default="./lumina_mgpt/ckpts/chameleon/tokenizer/vqgan.yaml",
        help="Path to the VQ model YAML config.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default="./lumina_mgpt/ckpts/chameleon/tokenizer/vqgan.ckpt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--remap", type=Path, default=None, help="Optional path to remap numpy file.")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional path to mapping numpy file.")
    parser.add_argument("--image", type=Path, required=True, help="Input image file to encode/decode.")
    parser.add_argument(
        "--output-image",
        type=Path,
        required=True,
        help="Directory or file path whose location will receive the generated comparison images.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Optional device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--palette-seed",
        type=int,
        default=0,
        help="Random seed for generating the latent token color palette.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    label, _ = _determine_remap_label(args.remap)
    output_dir = _ensure_output_directory(args.output_image)

    compressed_recon_path = output_dir / f"recon_{label}.png"
    compressed_latent_path = output_dir / f"latent_{label}.png"
    baseline_recon_path = output_dir / "recon_full.png"
    baseline_latent_path = output_dir / "latent_full.png"

    #mapping+remap 
    compressed_model = _load_model(args.config, args.ckpt, args.device)  # 不用內建 remap

    tokens, h, w = encode_image(compressed_model, args.image)  # 原始 codebook index

    if args.mapping is None or args.remap is None:
        raise ValueError("必須同時提供 --mapping 與 --remap 檔案路徑")

    mapping = np.load(args.mapping)
    remap = np.load(args.remap)

    # 1. 原始 code index → cluster id
    print(f"Encoded tokens: {tokens}")
    cluster_ids = mapping[tokens.cpu().numpy()]
    print(f"Map to cluster seq: {cluster_ids}")
    # 2. cluster id → 代表 code index
    representative_tokens = remap[cluster_ids]
    print(f"Remap to representative tokens: {representative_tokens}")
    representative_tokens = torch.from_numpy(representative_tokens).to(tokens.device)
    different = (tokens != representative_tokens).sum().item()
    print(f"Different amount of tokens: {different} / {tokens.numel()}")
    latent_image = _save_latent_visualization(
        tokens=representative_tokens,
        height=h,
        width=w,
        output_path=compressed_latent_path,
        palette_seed=args.palette_seed,
    )

    recon = decode_tokens(compressed_model, representative_tokens, h, w)
    recon_img = _tensor_to_image(recon)
    compressed_recon_path.parent.mkdir(parents=True, exist_ok=True)
    recon_img.save(compressed_recon_path)
    LOG.info("Reconstruction written to %s", compressed_recon_path)

    if args.remap is not None:
        baseline_model = _load_model(args.config, args.ckpt, args.device)
        baseline_tokens, bh, bw = encode_image(baseline_model, args.image)

        baseline_latent_image = _save_latent_visualization(
            tokens=baseline_tokens,
            height=bh,
            width=bw,
            output_path=baseline_latent_path,
            palette_seed=args.palette_seed,
        )

        baseline_recon = decode_tokens(baseline_model, baseline_tokens, bh, bw)
        baseline_img = _tensor_to_image(baseline_recon)
        baseline_recon_path.parent.mkdir(parents=True, exist_ok=True)
        # baseline_img.save(baseline_recon_path)
        LOG.info("Baseline reconstruction (full codebook) written to %s", baseline_recon_path)

        comparison_path = output_dir / f"comparison_{label}.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(np.asarray(recon_img))
        axes[0, 0].set_title(f"Reconstruction ({label})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(np.asarray(baseline_img))
        axes[0, 1].set_title("Reconstruction (full)")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(np.asarray(latent_image))
        axes[1, 0].set_title(f"Latent tokens ({label})")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(np.asarray(baseline_latent_image))
        axes[1, 1].set_title("Latent tokens (full)")
        axes[1, 1].axis("off")

        fig.tight_layout()
        fig.savefig(comparison_path, dpi=200)
        plt.close(fig)
        LOG.info("Saved comparison grid to %s", comparison_path)


if __name__ == "__main__":
    main()
