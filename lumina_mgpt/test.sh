# !/bin/bash
root_dir=results/kodak_gen_by_confidence
input_folder=/home/kris/generation_for_compression/dataset/image_kodak
mkdir -p ${root_dir}
for confidence in 0.35 0.37;  do
    mkdir -p ${root_dir}/conf_${confidence}
    python test.py \
        --allow_generation \
        --generate_mode confidence-based \
        --generate_conf_thres $confidence \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/conf_${confidence}/result.csv \
        --output_folder ${root_dir}/conf_${confidence}
done

