#!/bin/bash
ckpt_path=./ckpts/chameleon/tokenizer/vqgan.ckpt
config_path=./ckpts/chameleon/tokenizer/vqgan.yaml
method=dkm
output_dir=./codebook_remap_$method

for k_values in 4096 2048 1024 512 256 128; do  
    # python ./model/chameleon_vae_ori/compress_codebook.py \
    #     --method $method \
    #     --ckpt $ckpt_path \
    #     --output-dir $output_dir \
    #     --k-values $k_values \
    #     --dkm-iter 50 \
    #     --dkm-init kmeans++ \
    #     --tau 0.1 \
    #     --dkm-verbose \
    #     --histogram \
    #     --write-mapping \
    #     --summary \
    #     --seed 32
    for i in $(seq -w 1 24); do
        python ./model/chameleon_vae_ori/remap_inference_tool.py \
            --config $config_path \
            --ckpt $ckpt_path \
            --mapping $output_dir/mapping_${method}_k${k_values}.npy \
            --remap $output_dir/remap_${method}_k${k_values}.npy \
            --image /home/kris/generation_for_compression/dataset/image_kodak/kodim${i}.png \
            --output-image $output_dir/vis/kodim${i}
    done
   

done