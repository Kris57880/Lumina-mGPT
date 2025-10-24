#!/bin/bash
input_folder=../../dataset/image_kodak
output_dir=./exps/different_k_and_input_sizes_kodak
mkdir -p $output_dir
# Baseline 
# mkdir -p $output_dir/k=full
# python batch_image_codec.py \
#     --input-dir $input_folder \
#     --output-dir $output_dir/k=full \
#     --csv $output_dir/k=full/summary.csv \

# Different k
# for k in 4096 2048 1024 512; do
#     mkdir -p $output_dir/k=${k}
#     python batch_image_codec.py \
#     --input-dir $input_folder \
#     --output-dir $output_dir/k=${k} \
#     --csv $output_dir/k=${k}/summary.csv \
#     --mapping ./codebook_remap_dkm/mapping_dkm_k${k}.npy \
#     --remap ./codebook_remap_dkm/remap_dkm_k${k}.npy
# done

# Different input sizes and K 
for size in  384 256 768 1024; do
    for k in 4096 2048 1024 512; do
        mkdir -p $output_dir/${size}_k=${k}
        python batch_image_codec.py \
        --input-dir $input_folder \
        --output-dir $output_dir/${size}_k=${k} \
        --csv $output_dir/${size}_k=${k}/summary.csv \
        --mapping ./codebook_remap_dkm/mapping_dkm_k${k}.npy \
        --remap ./codebook_remap_dkm/remap_dkm_k${k}.npy \
        --input-size ${size}
    done
done