from inference_solver import FlexARInferenceSolver
from PIL import Image
import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import csv
import argparse


from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)

def compute_rate(likelihoods):
    bits = (likelihoods.log().sum()) / -math.log(2.0)

    return bits

def compute_psnr(img1, img2):
    """計算PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=False)

def pil_to_tensor(pil_image):
    """將PIL圖像轉換為tensor"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(pil_image).unsqueeze(0)



# Parse command line arguments
parser = argparse.ArgumentParser(description='Test script for image compression with Lumina-mGPT')
parser.add_argument('--allow_generation', action='store_true', help='Allow generation in compression mode')
parser.add_argument('--generate_mode', type=str, default="None", help='Mode for generation in compression')
parser.add_argument('--generate_prob', type=float, default=0, help='Probability for generation in compression')
parser.add_argument('--generate_conf_thres', type=float, default=1, help='Confidence threshold for generation in compression')
parser.add_argument('--input_folder', type=str, default="/home/kris/generation_for_compression/dataset/image_kodak", help='Path to input folder containing images')
parser.add_argument('--output_csv', type=str, default="kodak_results.csv", help='Output CSV file name')
parser.add_argument('--output_folder', type=str, default="results/kodak_gen", help='Output folder for generated images and results')

args = parser.parse_args()

inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768-Omni",
    # model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
    precision="fp16",
    target_size=768,
)

inference_solver.model.logits_processor = inference_solver.create_logits_processor(cfg=1.0, image_top_k=200)
inference_solver.model.compression = True
inference_solver.model.allow_generation = args.allow_generation
inference_solver.model.gen_mode = args.generate_mode
inference_solver.model.gen_prob = args.generate_prob  # set generation probability
inference_solver.model.gen_conf_thres = args.generate_conf_thres

# print model parameters size
# model_size = sum(p.numel() for p in inference_solver.model.parameters() if p.requires_grad)
# print("AR Model Parameters:")
# print(f"Model size: {model_size / 1e6:.2f}M parameters")
# print(f"Model size: {model_size / 1e9:.2f}B parameters")

# model_size = sum(p.numel() for p in inference_solver.item_processor.chameleon_ori_image_tokenizer._vq_model.parameters() if p.requires_grad)
# print("Tokenizer parameters:")
# print(f"Model size: {model_size / 1e6:.2f}M parameters")
# print(f"Model size: {model_size / 1e9:.2f}B parameters")


# 設定輸入和輸出路徑
input_folder = args.input_folder
output_csv = args.output_csv
output_folder = args.output_folder

image_limit = -1 

# 創建輸出文件夾
os.makedirs(output_folder, exist_ok=True)

# 準備結果列表
results = []
all_img_tok_distri_entropy = []
all_img_tok_entropy = []
all_img_time = []
if image_limit > 0:
    img_list = sorted(os.listdir(input_folder))[:image_limit]
else:
    img_list = sorted(os.listdir(input_folder))
print(f"共找到 {len(img_list)} 張圖片")

# img_list = ["kodim04.png"]
# print("測試用，只使用一張圖片")


print(f"開始處理圖片，請稍候...")


for i, image_name in enumerate(img_list):
    image_path = os.path.join(input_folder, image_name)
    
    if not os.path.exists(image_path):
        print(f"警告: {image_path} 不存在，跳過")
        continue
    
    # 檢查圖片是否存在
    if not os.path.isfile(image_path):
        print(f"警告: {image_path} 不是一個有效的文件，跳過")
        continue
    
    # 檢查圖片格式是否為PNG或JPEG
    if not (image_name.lower().endswith('.png') or image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg')):
        print(f"警告: {image_name} 不是PNG或JPEG格式，跳過")
        continue
    
    print(f"處理第 {i+1} 張圖片: {image_name}")
    

    # 載入圖片
    original_image = Image.open(image_path)
    images = [original_image]
    qas = [["", None]]

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Record the start event before the GPU operation
    start_event.record()
    torch.cuda.synchronize() 

    # 進行壓縮
    generated = inference_solver.compress(
        images=images,
        qas=qas,
        max_gen_len=2352+7,
        temperature=1.0,
        logits_processor=inference_solver.create_logits_processor(cfg=1.0, image_top_k=200),
    )
    
    torch.cuda.synchronize() # Ensures the "kernel" completes before recording the end event
    end_event.record()
    torch.cuda.synchronize()

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)

    new_image = generated[1][0]

    
    # 將原圖和新圖轉換為tensor進行比較
    original_image_tensor = pil_to_tensor(original_image)
    new_image_tensor = pil_to_tensor(new_image)
    gen_h, gen_w = new_image_tensor.shape[2], new_image_tensor.shape[3]
    if new_image_tensor.size() != original_image_tensor.size():
        print(f"注意: 重建圖片尺寸 {new_image_tensor.size()} 與原圖尺寸 {original_image_tensor.size()} 不符，正在調整尺寸...")
        # 將新圖調整到指定的輸出解析度
        new_image_tensor = F.interpolate(new_image_tensor, size=original_image_tensor.shape[2:], mode='bilinear', align_corners=False)
        new_image = transforms.ToPILImage()(new_image_tensor.squeeze(0))
    # 保存重建圖片
    output_image_path = os.path.join(output_folder, image_name)
    new_image.save(output_image_path)
    
    
    likelihoods = torch.tensor(inference_solver.model.likelihoods, dtype=torch.float32)
    tokens_to_transmit = torch.tensor(inference_solver.model.token_to_transmit, dtype=torch.int16)
    # First token and special tokens
    likelihoods[:3] = likelihoods[-1] = 1.0

    # 8803 is the End-of-Line token
    likelihoods[torch.where(tokens_to_transmit == 8803)] = 1.0
    tok_entropy= -torch.log2(likelihoods)
    tok_distri_entropy = torch.tensor(inference_solver.model.entropy_per_tok_distribution, dtype=torch.float32)

    all_img_tok_distri_entropy.append(tok_distri_entropy)
    all_img_tok_entropy.append(tok_entropy)
    all_img_time.append(elapsed_time_ms / 1000.0)  # Convert to seconds

    # 計算壓縮率
    total_bits = compute_rate(likelihoods)
    total_bits += 13 # First Token
    bpp = total_bits / (original_image_tensor.shape[2] * original_image_tensor.shape[3])
    
    # 計算圖像品質指標
    psnr_score = compute_psnr(original_image_tensor, new_image_tensor)
    ms_ssim_score = msssim_metric(original_image_tensor, new_image_tensor)
    lpips_score = lpips_metric(original_image_tensor * 2 - 1., new_image_tensor * 2 - 1.)
    
    # 記錄結果
    result = {
        'image_name': image_name,
        'total_bits': total_bits.item(),
        'bpp': bpp.item(),
        'psnr': psnr_score,
        'ms_ssim': ms_ssim_score.item(),
        'lpips': lpips_score.item(),
        'elapsed_time_sec': elapsed_time_ms / 1000.0,
    }
    results.append(result)
    # 將每個token的distribuion entropy likelihood 與繪製成圖表
    plot_figure = True 
    if plot_figure:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(25, 8))
        plt.rcParams.update({'font.size': 16})  
        plt.subplot(2, 1 , 1)
        plt.plot(tok_distri_entropy.numpy(), label='Entropy per Token')
        plt.xlabel('Token Index')
        plt.ylabel('Entropy (bits)')
        plt.title(f'Entropy Distribution for {image_name}')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1 , 2)
        plt.plot(tok_entropy.numpy(), label='Entropy per Token', color='orange')
        plt.xlabel('Token Index')
        plt.ylabel('Entropy (bits)')
        plt.title(f'Entropy each tokens for {image_name}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{image_name}_entropy_distribution.png"))
        plt.close()
        # 繪製散佈圖
        plt.figure(figsize=(15, 15))
        plt.scatter(tok_distri_entropy.numpy(), tok_entropy.numpy(), alpha=0.5)
        plt.xlabel('Distribution Entropy (bits)')
        plt.ylabel('Token Predict Entropy(bits)')
        plt.title(f'Entropy Scatter Plot for {image_name}')
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"{image_name}_entropy_scatter.png"))
        plt.close()
        # plot is_generated tokens as a square figure
        patch_size = 32
        h_grids = gen_h // patch_size
        w_grids = gen_w // patch_size
        latent_h = h_grids * 2
        latent_w = w_grids * 2
        if args.allow_generation: # plot where tokens are generated instead of transmitted
            is_generated = inference_solver.model.is_generated[3:3+latent_h * (latent_w +1)]# 3 = 1 Start token + 2 special tokens (H,W)
            is_generated = torch.tensor([1 if x else 0 for x in is_generated])
            is_generated = is_generated.view(latent_h, latent_w+1)
            plt.figure(figsize=(12, 10))
            plt.imshow(is_generated.numpy(), cmap='gray', aspect='auto')
            plt.colorbar(label='Generated Token (1=True, 0=False)')
            plt.title(f'Generated Tokens for {image_name}')
            plt.savefig(os.path.join(output_folder, f"{image_name}_generated_tokens.png"))
            plt.close()
        # plot the confidence scores as a heatmap
        tok_distri_entropy_plot = tok_distri_entropy[3:3+latent_h * (latent_w +1)].view(latent_h, latent_w+1) 
        tok_entropy_plot = tok_entropy[3:3+latent_h * (latent_w +1)].view(latent_h, latent_w+1)
        plt.figure(figsize=(8, 12))
        plt.subplot(2, 1, 1)
        plt.imshow(tok_distri_entropy_plot.numpy(), cmap='hot', aspect='auto')
        plt.colorbar(label='Token Distribution Entropy (bits)')
        plt.title(f'Token Distribution Entropy Heatmap for {image_name}')
        plt.subplot(2, 1, 2)
        plt.imshow(tok_entropy_plot.numpy(), cmap='hot', aspect='auto')
        plt.colorbar(label='Token Predict Entropy (bits)')
        plt.title(f'Token Predict Entropy Heatmap for {image_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{image_name}_token_entropy_heatmap.png"))
        plt.close()




    print(f" Save entropy distribution figure to {os.path.join(output_folder, f'{image_name}_entropy_distribution.png')}")
    print(f"  結果: BPP={bpp.item():.4f}, PSNR={psnr_score:.4f}dB, MS-SSIM={ms_ssim_score:.4f}, LPIPS={lpips_score:.4f}")


# 將結果保存到CSV文件
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_name', 'total_bits', 'bpp', 'psnr', 'ms_ssim', 'lpips', 'elapsed_time_sec']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)

# 計算平均值
if results:
    avg_bpp = sum(r['bpp'] for r in results) / len(results)
    avg_psnr = sum(r['psnr'] for r in results) / len(results)
    avg_ms_ssim = sum(r['ms_ssim'] for r in results) / len(results)
    avg_lpips = sum(r['lpips'] for r in results) / len(results)
    total_time = sum(r['elapsed_time_sec'] for r in results)
    avg_time = total_time / len(results)

    print(f"\n=== 統計結果 (共處理 {len(results)} 張圖片) ===")
    print(f"平均 BPP: {avg_bpp:.4f}")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 MS-SSIM: {avg_ms_ssim:.4f}")
    print(f"平均 LPIPS: {avg_lpips:.4f}")
    print(f"平均 每張圖片處理時間: {avg_time:.4f} 秒")
    print(f"總處理時間: {total_time:.4f} 秒")
    
    # 將平均值也保存到CSV
    avg_result = {
        'image_name': 'AVERAGE',
        'total_bits': 0,
        'bpp': avg_bpp,
        'psnr': avg_psnr,
        'ms_ssim': avg_ms_ssim,
        'lpips': avg_lpips,
        'elapsed_time_sec': avg_time,
    }
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(avg_result)
if plot_figure:
    # scatter plot entropy distribution vs token entropy for all images
    all_tok_distri_entropy = torch.cat(all_img_tok_distri_entropy)
    all_tok_entropy = torch.cat(all_img_tok_entropy)
    plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 16})  # 全部字體大小
    plt.scatter(all_tok_distri_entropy.numpy(), all_tok_entropy.numpy(), alpha=0.5)
    plt.xlabel('Distribution Entropy (bits)')
    plt.ylabel('Token Predict Entropy (bits)')
    plt.title(f'Entropy Scatter Plot for All Images')
    plt.grid()
    plt.savefig(os.path.join(output_folder, f"all_images_entropy_scatter.png"))
    plt.close()
    print(f" Save overall entropy scatter figure to {os.path.join(output_folder, f'all_images_entropy_scatter.png')}")

print(f"\n結果已保存到: {output_csv}")
print(f"重建圖片保存在: {output_folder}/")
