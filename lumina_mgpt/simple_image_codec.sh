#!/bin/bash

input_image=/home/kris/generation_for_compression/Lumina-mGPT/lumina_mgpt/exps/debug/prompt_1.png
output_dir=./exps/debug
mkdir -p $output_dir
python conditional_image_codec.py \
    -i $input_image \
    -o $output_dir/cat.png \
    -l $output_dir\
    --gen-prob 1\
    --prompt "Generate an image of 512x512 according to the following prompt:\nA photo of a cat sitting on a skateboard in the middle of the street"

# input_dir=/home/kris/generation_for_compression/dataset/image_kodak
# output_dir=./exps/generative_coding/gen_0.3
# mkdir -p "$output_dir"

# for input_image in "$input_dir"/*.png; do
#     [ -e "$input_image" ] || continue
#     base_name=$(basename "$input_image" .png)
#     echo "Processing $input_image ..."
#     python simple_image_codec.py \
#         -i "$input_image" \
#         -o "$output_dir/${base_name}.png" \
#         -l "$output_dir"\
#         --gen-prob 0.3

#     mv "$output_dir/per_entropy.csv" "$output_dir/per_image_entropy_${base_name}.csv"
#     mv "$output_dir/entropy.png" "$output_dir/${base_name}_entropy.png"
# done 
# for k in 2048; do
#     mkdir -p $output_dir/k=${k}
#     python simple_image_codec.py \
#     -i $input_image \
#     -o $output_dir/kodim04_compressed_k${k}.png \
#     -l $output_dir/k=${k} \
#     --mapping ./codebook_remap_dkm/mapping_dkm_k${k}.npy \
#     --remap ./codebook_remap_dkm/remap_dkm_k${k}.npy
# done
