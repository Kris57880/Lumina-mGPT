import os
import csv
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from inference_solver import FlexARInferenceSolver
import argparse


inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
    precision="bf16",
    target_size=768,
)

def load_captions(csv_path: str) -> List[Tuple[str, str]]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到 caption 檔案: {csv_file}")
    captions: List[Tuple[str, str]] = []
    with csv_file.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            caption = row.get("caption", "").strip()
            if not caption:
                continue
            image_name = row.get("image_name", "").strip()
            if not image_name:
                image_name = f"generated_{idx:04d}.png"
            elif not Path(image_name).suffix:
                image_name = f"{image_name}.png"
            captions.append((image_name, caption))
    if not captions:
        raise ValueError(f"未在 {csv_file} 找到任何有效 caption")
    return captions

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from captions CSV")
    parser.add_argument('-i', "--input_caption_file", required=True, help="輸入 caption CSV 檔路徑")
    parser.add_argument('-o', "--output_folder", default="generated_images", help="輸出圖片資料夾")
    args = parser.parse_args()

    caption_items = load_captions(args.input_caption_file)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_name, caption in caption_items:
        q = (
            "Generate an image of 768x768 according to the following prompt:\n"
            f"{caption}"
        )
        generated = inference_solver.generate(
            images=[],
            qas=[[q, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
        )
        images = generated[1] if len(generated) > 1 else []
        if not images:
            print(f"警告: {image_name} 未生成圖片，略過")
            continue
        new_image = images[0]
        new_image.save(output_dir / image_name)
        print(f"已產生: {output_dir / image_name}")

if __name__ == "__main__":
    main()
