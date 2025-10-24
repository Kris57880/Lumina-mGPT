#!/usr/bin/env python3
"""Batch image compression and entropy estimation built on Chameleon VAE."""

import csv
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from model.chameleon import ChameleonForConditionalGeneration  # type: ignore
from simple_image_codec import (  # type: ignore
    build_bpe_cluster_lookup,
    decoder,
    encoder,
    estimate_entropy_bits,
    init_tokenizer,
    load_remap_lookup,
    process_image,
)


def iter_images(input_dir: Path) -> Iterable[Path]:
    # Iterate over supported image files within the input directory.
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_exts:
            yield path


def process_directory(
    input_dir: Path,
    output_dir: Path,
    csv_path: Path,
    device: str,
    model_name: str,
    remap_lookup: Optional[np.ndarray],
    entropy_only: bool,
    input_size: Optional[int] = 512,
    gen_probabilities: float = 0.0,
) -> None:
    # Process all images within a directory, saving reconstructions and entropy metrics.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_tokenizer, vocab_info, vocab_translation = init_tokenizer(device)

    model: Optional[ChameleonForConditionalGeneration] = None
    if model_name:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        model.to(device)
        model.eval()

    cluster_lookup = build_bpe_cluster_lookup(remap_lookup, vocab_translation)
    cluster_lookup = cluster_lookup if cluster_lookup else None

    rows = [("filename", "tokens", "entropy_bits", "bpp")]

    for image_path in iter_images(input_dir):
        filename, tokens, total_bits, bpp = process_image(
            image_path,
            output_dir=output_dir,
            device=device,
            model=model,
            image_tokenizer=image_tokenizer,
            vocab_info=vocab_info,
            vocab_translation=vocab_translation,
            remap_lookup=remap_lookup,
            cluster_lookup=cluster_lookup,
            entropy_only=entropy_only,
            input_size=input_size,
            gen_probabilities=gen_probabilities,
        )
        rows.append((filename, tokens, total_bits, bpp))

    with csv_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nSummary written to CSV: {csv_path}")


def main() -> None:
    # Parse CLI arguments and launch batch compression workflow.
    parser = ArgumentParser(description="Batch image codec entropy analysis")
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for reconstructions")
    parser.add_argument("--csv", default="entropy_summary.csv", help="Output CSV path")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument(
        "--entropy-model",
        default="Alpha-VLLM/Lumina-mGPT-7B-768",
        help="Model checkpoint used for entropy estimation",
    )
    parser.add_argument("--mapping", default=None, help="Path to mapping.npy file")
    parser.add_argument("--remap", default=None, help="Path to remap.npy file")
    parser.add_argument(
        "--entropy-only",
        action="store_true",
        help="Skip reconstructions and only compute entropy statistics",
    )
    parser.add_argument(
        "--input-size", 
        type=int,
        default=512,
        help="Input image size",
    )
    parser.add_argument(
        '--gen-prob',
        type=float,
        default=0.0,
        help="Probability of using generation mode during encoding and ar",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found - {input_dir}")
        return

    remap_lookup = load_remap_lookup(args.mapping, args.remap)

    process_directory(
        input_dir=input_dir,
        output_dir=Path(args.output_dir),
        csv_path=Path(args.csv),
        device=args.device,
        model_name=args.entropy_model,
        remap_lookup=remap_lookup,
        entropy_only=args.entropy_only,
        input_size=args.input_size,
        gen_probabilities=args.gen_prob,
    )


if __name__ == "__main__":
    main()
