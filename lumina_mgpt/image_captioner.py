import os
import csv
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from inference_solver import FlexARInferenceSolver


class ImageCaptioner:
    """
    用於批量生成圖片詳細描述的工具類
    """
    
    def __init__(self, model_path: str = "Alpha-VLLM/Lumina-mGPT-7B-512", 
                 precision: str = "bf16", 
                 target_size: int = 512):
        """
        初始化 ImageCaptioner
        
        Args:
            model_path: 模型路徑
            precision: 精度設置 ("bf16", "fp16", "fp32")
            target_size: 目標圖片尺寸
        """
        self.inference_solver = FlexARInferenceSolver(
            model_path=model_path,
            precision=precision,
            target_size=target_size,
        )
        self.prompt = (
            "You are an image-to-prompt generator. Analyze the image <|image|> carefully and create a short prompt suitable for image generation models. "
            "Only include details that are visually present in the image — do NOT guess, imagine, or add unseen elements. "
            "start with 'Image of ...'"
            "Use concise, comma-separated keywords, including subject, medium/style, lighting, mood, color tone, and composition."
        )

    def get_supported_image_formats(self) -> List[str]:
        """
        返回支持的圖片格式
        """
        return ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """
        從資料夾中獲取所有圖片檔案路徑
        
        Args:
            folder_path: 資料夾路徑
            
        Returns:
            圖片檔案路徑列表
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"資料夾不存在: {folder_path}")
        
        if not folder.is_dir():
            raise NotADirectoryError(f"路徑不是資料夾: {folder_path}")
        
        supported_formats = self.get_supported_image_formats()
        image_files = []
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def generate_caption(self, image_path: str) -> str:
        """
        為單個圖片生成詳細描述
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            生成的圖片描述
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 使用 inference_solver 生成描述
            generated = self.inference_solver.generate(
                images=[image],
                qas=[[self.prompt, None]],
                max_gen_len=8192,
                temperature=1.0,
                logits_processor=self.inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
            )
            
            caption = generated[0]
            return caption
        
        except Exception as e:
            print(f"處理圖片 {image_path} 時發生錯誤: {str(e)}")
            return ""
    
    def process_folder(self, folder_path: str, output_csv: str = None) -> str:
        """
        批量處理資料夾中的所有圖片，生成描述並存成 CSV 檔案
        
        Args:
            folder_path: 輸入資料夾路徑
            output_csv: 輸出 CSV 檔案路徑（默認為 folder_name_captions.csv）
            
        Returns:
            輸出 CSV 檔案路徑
        """
        # 取得所有圖片檔案
        image_files = self.get_image_files(folder_path)
        
        if not image_files:
            print(f"在 {folder_path} 中未找到圖片檔案")
            return None
        
        print(f"找到 {len(image_files)} 個圖片檔案")
        
        # 如果未指定輸出路徑，根據資料夾名稱生成
        if output_csv is None:
            folder_name = Path(folder_path).name
            output_csv = f"{folder_name}_captions.csv"
        
        # 生成描述並寫入 CSV
        results = []
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] 正在處理: {image_path}")
            
            caption = self.generate_caption(image_path)
            
            results.append({
                'image_path': image_path,
                'image_name': Path(image_path).name,
                'caption': caption
            })
            
            print(f"Image {image_path}, caption: {caption}")
        
        # 將結果寫入 CSV 檔案
        self.save_to_csv(results, output_csv)
        
        return output_csv
    
    @staticmethod
    def save_to_csv(results: List[dict], output_path: str) -> None:
        """
        將結果保存為 CSV 檔案
        
        Args:
            results: 包含圖片路徑和描述的字典列表
            output_path: 輸出 CSV 檔案路徑
        """
        if not results:
            print("沒有結果要保存")
            return
        
        keys = results[0].keys()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n結果已保存到: {output_path}")


def main():
    """
    主函數 - 示例用法
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="批量生成圖片詳細描述")
    parser.add_argument("-i", "--input", help="輸入圖片資料夾路徑")
    parser.add_argument("-o", "--output", help="輸出 CSV 檔案路徑（可選）")
    parser.add_argument("--model-path", default="Alpha-VLLM/Lumina-mGPT-7B-512", 
                        help="模型路徑")
    parser.add_argument("--precision", default="bf16", 
                        help="精度設置 (bf16/fp16/fp32)")
    parser.add_argument("--target-size", type=int, default=512, 
                        help="目標圖片尺寸")
    
    args = parser.parse_args()
    
    # 初始化 captioner
    captioner = ImageCaptioner(
        model_path=args.model_path,
        precision=args.precision,
        target_size=args.target_size
    )
    
    # 處理資料夾
    output_file = captioner.process_folder(args.input, args.output)
    
    if output_file:
        print(f"\n✓ 所有圖片描述已生成完畢")
        print(f"  輸出檔案: {output_file}")


if __name__ == "__main__":
    main()
