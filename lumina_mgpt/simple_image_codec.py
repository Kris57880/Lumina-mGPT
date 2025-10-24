#!/usr/bin/env python3
"""Minimal image codec built on the Chameleon VAE."""

import math
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from model.chameleon import ChameleonForConditionalGeneration  # type: ignore
from tqdm import trange

# Add project root to the Python path
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from model.chameleon_vae_ori import ImageTokenizer, VocabInfo, VocabTranslation  # type: ignore


def load_remap_lookup(mapping_path: Optional[str], remap_path: Optional[str]) -> Optional[np.ndarray]:
    # Load mapping/remap files and produce an original → representative lookup.
    if mapping_path is None and remap_path is None:
        return None
    if not mapping_path or not remap_path:
        raise ValueError("Both mapping and remap files must be provided together")
    mapping = np.load(mapping_path)
    remap = np.load(remap_path)
    if mapping.ndim != 1 or remap.ndim != 1:
        raise ValueError("Expected one-dimensional mapping and remap arrays")
    mapping = mapping.astype(np.int64)
    remap = remap.astype(np.int64)
    if mapping.max(initial=0) >= remap.shape[0]:
        raise ValueError("Mapping values exceed remap length; files likely mismatch")
    return remap[mapping]


def map_tokens_to_representatives(tokens: torch.Tensor, remap_lookup: Optional[np.ndarray]) -> torch.Tensor:
    # Replace tokens with their representative IDs using the lookup array.
    if remap_lookup is None:
        return tokens
    flat = tokens.view(-1).cpu().numpy().astype(np.int64)
    if flat.size == 0:
        return tokens
    if flat.max(initial=0) >= remap_lookup.shape[0]:
        raise ValueError("Token ID exceeds remap lookup length")
    mapped = remap_lookup[flat]
    rep_tensor = torch.from_numpy(mapped).to(tokens.device)
    return rep_tensor.view_as(tokens)


def build_bpe_cluster_lookup(remap_lookup: Optional[np.ndarray], vocab_translation: VocabTranslation) -> Dict[int, torch.Tensor]:
    # Build mapping from representative BPE IDs to the member BPE IDs in the same cluster.
    if remap_lookup is None:
        return {}
    cluster_members: Dict[int, list[int]] = {}
    img2bpe = vocab_translation.img2bpe
    for original_id, representative_id in enumerate(remap_lookup.tolist()):
        if original_id not in img2bpe or representative_id not in img2bpe:
            continue
        rep_bpe = int(img2bpe[representative_id])
        orig_bpe = int(img2bpe[original_id])
        cluster_members.setdefault(rep_bpe, []).append(orig_bpe)
    return {key: torch.tensor(ids, dtype=torch.long) for key, ids in cluster_members.items()}


def flatten_bpe_tokens(bpe_tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    # Flatten BPE tokens and infer latent grid dimensions.
    if bpe_tokens.dim() == 2:
        h_latent, w_latent = bpe_tokens.shape
        flat_tokens = bpe_tokens.flatten()
    else:
        flat_tokens = bpe_tokens
        total_tokens = len(flat_tokens)
        latent_dim = int(total_tokens ** 0.5)
        h_latent = w_latent = latent_dim
    return flat_tokens, h_latent, w_latent


def encoder(
    input_path: str,
    image_tokenizer: ImageTokenizer,
    vocab_info: VocabInfo,
    vocab_translation: VocabTranslation,
    remap_lookup: Optional[np.ndarray],
    size: Optional[int] = None,
) -> dict:
    # Encode an image into representative image and BPE tokens, returning flattened tokens and metadata.
    image = Image.open(input_path).convert("RGB")
    num_pixels = image.size[0] * image.size[1] # record original size
    if size is not None:
        image = image.resize((size, size), Image.BICUBIC)

    image_tokens = image_tokenizer.img_tokens_from_pil(image)
    image_tokens = map_tokens_to_representatives(image_tokens, remap_lookup)
    bpe_tokens = vocab_translation.convert_img2bp2(image_tokens)
    flat_tokens, h_latent, w_latent = flatten_bpe_tokens(bpe_tokens)

    flat_tokens_cpu = flat_tokens.to(torch.int64).cpu()
    sequence = torch.cat(
        (
            torch.tensor([vocab_info.begin_sequence], dtype=torch.int64),
            flat_tokens_cpu,
            torch.tensor([vocab_info.end_sequence], dtype=torch.int64),
        )
    )

    return {
        "flat_bpe_tokens": flat_tokens_cpu,
        "h_latent": h_latent,
        "w_latent": w_latent,
        "num_pixels": num_pixels,
        "sequence": sequence,
    }


def load_vocab_info(vocab_path: str) -> VocabInfo:
    # Load vocabulary metadata from disk.
    with open(vocab_path, encoding="utf8") as f:
        vocab_data = json.load(f)
    return VocabInfo(vocab_data["model"]["vocab"])


def init_tokenizer(device: str):
    # Initialize tokenizer, vocab info, and translation helpers for images.
    image_tokenizer = ImageTokenizer(
        cfg_path="./ckpts/chameleon/tokenizer/vqgan.yaml",
        ckpt_path="./ckpts/chameleon/tokenizer/vqgan.ckpt",
        device=device,
    )
    vocab_info = load_vocab_info("./ckpts/chameleon/tokenizer/text_tokenizer.json")
    vocab_translation = VocabTranslation(vocab_info, device=device)
    return image_tokenizer, vocab_info, vocab_translation


def estimate_entropy_bits(
    bpe_tok_seq: torch.Tensor,
    model: ChameleonForConditionalGeneration,
    device: str,
    bpe_cluster_lookup: Optional[Dict[int, torch.Tensor]] = None,
    gen_probabilities: float = 0.0, # probability to use generation instead of compression
) -> tuple[float, list[float], list[float], torch.Tensor]:
    # Estimate cumulative and per-token entropy bits for the provided sequence.
    total_bits = 0.0
    per_token_bits: list[float] = []
    distribution_bits: list[float] = []
    new_bpe_tok_seq = bpe_tok_seq.clone()
    prefix = new_bpe_tok_seq[0:1]
    for idx in trange(1, bpe_tok_seq.numel()-1, ncols=80):#ignore the last token (which for end signal)
        input_ids = prefix.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            logits = logits[4:8195+1] # image tokens get from MultiModalLogitsProcessor 
        assert logits.shape[0] == 8192
        # print(logits.argmax(dim=-1))
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_float = log_probs.float()
        probs = torch.exp(log_probs_float)
        dist_entropy_bits = float(-(probs * log_probs_float).sum().item() / math.log(2))
        distribution_bits.append(dist_entropy_bits)
        if torch.rand(1).item() < gen_probabilities:
            # use generation, select the highest probability token
            bpe_tok = logits.argmax(dim=-1)
            new_bpe_tok_seq[idx] = bpe_tok # replace token to generated one
            bits = 0.0
        else:
            # use compression
            bpe_tok = int(new_bpe_tok_seq[idx].item())

            if bpe_cluster_lookup and bpe_tok in bpe_cluster_lookup:
                members = bpe_cluster_lookup[bpe_tok].to(log_probs.device)
                cluster_log_prob = torch.logsumexp(log_probs_float[members], dim=0)
                bits = float(-cluster_log_prob / math.log(2))
            else:
                token_log_prob = log_probs_float[bpe_tok-4].item() # bpe token to img_token
                bits = -token_log_prob / math.log(2)

        per_token_bits.append(bits)
        total_bits += bits
        prefix = new_bpe_tok_seq[0 : idx + 1]

    # last token is end token, not counted
    per_token_bits += [0.0]  # for the last token
    distribution_bits += [0.0]  # for the last token

    return total_bits, per_token_bits, distribution_bits, new_bpe_tok_seq


def decoder(
    flat_bpe_tokens: torch.Tensor,
    image_tokenizer: ImageTokenizer,
    vocab_translation: VocabTranslation,
    remap_lookup: Optional[np.ndarray],
    latent_shape: Tuple[int, int],
    device: str,
    output_path: str,
):
    # Decode representative image tokens back into an image and save to disk.
    h_latent, w_latent = latent_shape
    decoded_tokens: list[int] = []
    missing_tokens: set[int] = set()
    for token in flat_bpe_tokens:
        token_id = int(token.item())
        mapped = vocab_translation.bpe2img.get(token_id)
        if mapped is None:
            missing_tokens.add(token_id)
            mapped = token_id
        decoded_tokens.append(int(mapped))

    if missing_tokens:
        sample = sorted(missing_tokens)[:5]
        print(
            "[decoder] Warning: encountered BPE tokens without img mapping; "
            f"falling back to identity for {len(missing_tokens)} tokens (examples: {sample})"
        )

    img_tokens = torch.tensor(decoded_tokens, device=device)
    img_tokens = map_tokens_to_representatives(img_tokens, remap_lookup)

    reconstructed_image = image_tokenizer.pil_from_img_toks(
        img_tokens, h_latent, w_latent
    )
    reconstructed_image.save(output_path)

    return reconstructed_image


def analyze_entropy(
    sequence: torch.Tensor,
    vocab_translation: VocabTranslation,
    log_path: str,
    num_pixels: int,
    model: ChameleonForConditionalGeneration,
    device="cuda",
    remap_lookup: Optional[np.ndarray] = None,
    bpe_cluster_lookup: Optional[Dict[int, torch.Tensor]] = None,
    gen_probabilities: float = 0.0,
):
    # Compute entropy statistics from precomputed tokens and persist metrics to disk.
    os.makedirs(log_path, exist_ok=True)


    if bpe_cluster_lookup is None:
        bpe_cluster_lookup = build_bpe_cluster_lookup(remap_lookup, vocab_translation)
    
    if sequence.numel() < 2:
        raise ValueError("Sequence must include at least a BOS and EOS token")

    num_tokens = sequence.numel() - 1  # exclude BOS for logging/alignment
    total_bits, per_token_bits, distribution_bits, gen_seq = estimate_entropy_bits(
        sequence.to(device),
        model,
        device,
        bpe_cluster_lookup=bpe_cluster_lookup,
        gen_probabilities=gen_probabilities,
    )
    bpp = total_bits / max(num_pixels, 1) # bpp calculate using original image size
    
    print(f"Tokens:       {num_tokens-1}")
    print(f"Entropy bits: {total_bits:.4f}")
    print(f"Average per-token bits: {total_bits/(num_tokens-1):.4f}")
    print(f"Entropy bpp:  {bpp:.6f}")

    csv_path = os.path.join(log_path, "per_entropy.csv")
    png_path = os.path.join(log_path, "entropy.png")

    target_tokens = gen_seq[1 : 1 + num_tokens].cpu().numpy() # target tokens include EOS
    img_token_ids = [
        vocab_translation.bpe2img.get(int(token), int(token))
        for token in target_tokens
    ]

    per_token_bits_arr = np.array(per_token_bits[:num_tokens])
    distribution_bits_arr = np.array(distribution_bits[:num_tokens])

    log_steps = np.arange(1, num_tokens + 1)
    data = np.column_stack([
        log_steps,
        img_token_ids,
        per_token_bits_arr,
        distribution_bits_arr,
    ])
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="step,img_token_id,target_token_bits,distribution_entropy_bits",
        comments="",
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(25, 6), sharex=True)

    ax_top.plot(log_steps, per_token_bits, linewidth=1.2, color="tab:blue")
    ax_top.set_ylabel("Target token bits")
    ax_top.set_title("Per-token entropy (bits)")
    ax_top.grid(True, linestyle="--", alpha=0.3)

    ax_bottom.plot(log_steps, distribution_bits, linewidth=1.2, color="tab:orange")
    ax_bottom.set_ylabel("Distribution entropy (bits)")
    ax_bottom.set_xlabel("Token index")
    ax_bottom.set_title("Logits distribution entropy (bits)")
    ax_bottom.grid(True, linestyle="--", alpha=0.3)

    sqrt_tokens = math.sqrt(max(num_tokens, 1))
    num_ticks = max(1, int(math.floor(sqrt_tokens)))
    tick_pairs = []
    for k in range(1, num_ticks + 1):
        pos = min(num_tokens, max(1, int(round(k * sqrt_tokens))))
        if not tick_pairs or pos != tick_pairs[-1][0]:
            tick_pairs.append((pos, k))
    if tick_pairs:
        tick_positions, _ = zip(*tick_pairs)
        ax_bottom.set_xticks(tick_positions)
        ax_bottom.set_xticklabels([str(pos) for pos in tick_positions])

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    print(f"log file saved to: {log_path}")
    return total_bits, bpp, gen_seq[1 : -1]  # exclude BOS and EOS



def process_image(
    image_path: Path,
    *,
    output_dir: Path,
    device: str,
    model: Optional[ChameleonForConditionalGeneration],
    image_tokenizer,
    vocab_info,
    vocab_translation,
    remap_lookup: Optional[np.ndarray],
    cluster_lookup,
    entropy_only: bool,
    input_size: Optional[int],
    gen_probabilities: float = 0.0,
    log_path: str = "log",
) -> tuple[str, int, float, float]:
    # Compress, estimate entropy, and optionally reconstruct a single image.
    print(f"\n=== Processing {image_path.name} ===")
    encoded = encoder(
        input_path=str(image_path),
        image_tokenizer=image_tokenizer,
        vocab_info=vocab_info,
        vocab_translation=vocab_translation,
        remap_lookup=remap_lookup,
        size=input_size,
    )
    decode_bpe_tokens = encoded["flat_bpe_tokens"]
    token_count = int(encoded["flat_bpe_tokens"].numel())

    img_bits,bpp, gen_bpe_tokens = analyze_entropy(
        sequence=encoded["sequence"],
        vocab_translation=vocab_translation,
        log_path=log_path,
        num_pixels=encoded["num_pixels"],
        device=device,
        model=model,
        remap_lookup=remap_lookup,
        bpe_cluster_lookup=cluster_lookup,
        gen_probabilities=gen_probabilities,
    )
    if entropy_only:
        return image_path.name, token_count, img_bits, bpp
    
    decode_bpe_tokens = gen_bpe_tokens
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    decoder(
        flat_bpe_tokens=decode_bpe_tokens,
        image_tokenizer=image_tokenizer,
        vocab_translation=vocab_translation,
        remap_lookup=remap_lookup,
        latent_shape=(encoded["h_latent"], encoded["w_latent"]),
        device=device,
        output_path=os.path.join(output_dir, image_path.name),
    )

    return image_path.name, token_count, img_bits, bpp


def main():
    # Parse CLI arguments and run entropy analysis and/or codec.
    parser = ArgumentParser(description="Minimal image codec")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--log_path", "-l", default="entropy_log", help="Directory for entropy logs")
    parser.add_argument("--size", type=int, default=None, help="Resize input image to size x size")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument("--entropy-model",default="Alpha-VLLM/Lumina-mGPT-7B-768",help="Model checkpoint used for entropy estimation",)
    parser.add_argument("--entropy-only", action="store_true", help="Skip decoding and only compute entropy")
    parser.add_argument("--mapping", default=None, help="Path to mapping.npy file")
    parser.add_argument("--remap", default=None, help="Path to remap.npy file")
    parser.add_argument("--gen-prob", type=float, default=0.0, help="Probability of using generation instead of compression per token")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found - {args.input}")
        return

    remap_lookup = load_remap_lookup(args.mapping, args.remap)


    image_tokenizer, vocab_info, vocab_translation = init_tokenizer(args.device)

    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    # if args.entropy_model == "Alpha-VLLM/Lumina-mGPT-7B-768":
    #     model_path = "facebook/chameleon-7b"
    model = ChameleonForConditionalGeneration.from_pretrained(
        args.entropy_model , torch_dtype=torch_dtype
    )
    model.to(args.device)
    model.eval()

    process_image(
        Path(args.input),
        output_dir=Path(args.output).parent,
        device=args.device,
        model=model,
        image_tokenizer=image_tokenizer,
        vocab_info=vocab_info,
        vocab_translation=vocab_translation,
        remap_lookup=remap_lookup,
        cluster_lookup=build_bpe_cluster_lookup(remap_lookup, vocab_translation),
        entropy_only=args.entropy_only,
        input_size=args.size,
        gen_probabilities=args.gen_prob,
        log_path=args.log_path,
    )

if __name__ == "__main__":
    main()