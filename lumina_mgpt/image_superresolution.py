import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from inference_solver import FlexARInferenceSolver
import argparse


inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768-Omni",
    precision="bf16",
    target_size=768,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from low-resolution conditions")
    parser.add_argument('-i', "--input_folder", required=True, help="輸入影像資料夾路徑")
    parser.add_argument('-o', "--output_folder", default="generated_images", help="輸出圖片資料夾")
    parser.add_argument('--condition_size', type=int, default=256, help="條件影像邊長解析度")
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    if not input_dir.exists():
        raise FileNotFoundError(f"找不到輸入資料夾: {input_dir}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".png", ".jpg", ".jpeg"}
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in supported_exts:
            continue
        try:
            with Image.open(image_path) as img:
                source_image = img.convert("RGB")
        except Exception as exc:
            print(f"警告: {image_path} 開啟失敗，原因: {exc}")
            continue

        condition_image = source_image.resize(
            (args.condition_size, args.condition_size),
            Image.Resampling.BICUBIC,
        )
            
        q = (
            "Given this low-resolution image <|image|>, generate a ultra high-resolution, sharper, clear, and detailed version (768x768). "
            "Preserve the original content, composition, and colors. "
        )
        generated = inference_solver.generate(
            images=[condition_image],
            qas=[[q, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=7.0, image_top_k=2000),
        )
        images = generated[1] if len(generated) > 1 else []
        if not images:
            print(f"警告: {image_path.name} 未產生結果，略過")
            continue
        images[0].save(output_dir / image_path.name)
        print(f"已產生: {output_dir / image_path.name}")

if __name__ == "__main__":
    main()
