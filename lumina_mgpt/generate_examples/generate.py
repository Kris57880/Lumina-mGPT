import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse

from PIL import Image
import torch

from inference_solver import FlexARInferenceSolver
from xllmx.util.misc import random_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    args = parser.parse_args()

    print("args:\n", args)

    select_set1 = [
        "",
    ]

    l_prompts = select_set1

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
    )

    with torch.no_grad():
        l_generated_all = []
        for i, prompt in enumerate(l_prompts):
            for repeat_idx in range(n):
                random_seed(repeat_idx)
                generated = inference_solver.generate(
                    images=[Image.open("/home/kris/generation_for_compression/dataset/image_kodak/kodim04.png")],
                    qas=[[f"", None]],
                    max_gen_len=8192,
                    temperature=t,
                    logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                )
                try:
                    l_generated_all.append(generated[1][0])
                except:
                    l_generated_all.append(Image.new("RGB", (w, h)))
                    print(f"Warning: Generation failed for prompt {i}, repeat {repeat_idx}")

        result_image = inference_solver.create_image_grid(l_generated_all, len(l_prompts), n)
        result_image.save(args.save_path)
