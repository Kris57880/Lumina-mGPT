# !/bin/bash
input_folder=/home/kris/generation_for_compression/dataset/image_kodak
# root_dir=results/debug
# mkdir -p ${root_dir}
# python test.py \
#     --input_folder ${input_folder} \
#     --output_csv ${root_dir}/result.csv \
#     --output_folder ${root_dir} 



# root_dir=results/kodak_uncond_gen_compression
# mkdir -p ${root_dir}    
# exp_name=by_confidence_w_predifined_map
# mkdir -p ${root_dir}/${exp_name}
# for conf in 0.3 0.5 0.7; do 
#     mkdir -p ${root_dir}/${exp_name}/conf_${conf}
#     python test.py \
#         --input_folder ${input_folder} \
#         --output_csv ${root_dir}/${exp_name}/conf_${conf}/result.csv \
#         --output_folder ${root_dir}/${exp_name}/conf_${conf} \
#         --input_confidence_folder results/kodak_normal \
#         --allow_generation \
#         --generate_mode confidence-with-map \
#         --generate_conf_thres ${conf} 
# done 

# root_dir=results/kodak_uncond_gen_compression
# mkdir -p ${root_dir}
# exp_name=gen_every_n_tokens
# mkdir -p ${root_dir}/${exp_name}
# for n in 4 3 2 ; do
#     mkdir -p ${root_dir}/${exp_name}/n=${n}
#     python test.py \
#         --input_folder ${input_folder} \
#         --output_csv ${root_dir}/${exp_name}/n=${n}/result.csv \
#         --output_folder ${root_dir}/${exp_name}/n=${n} \
#         --allow_generation \
#         --generate_mode gen-every-n \
#         --gen_every_n_tokens ${n} 
# done

# different CFGs with small image as condition
# for res in 256; do
#     for cfg in 2.0 4.0 7.0; do 
#         root_dir=results/small_img_as_cond/kodak_img_cond_${res}_cfg${cfg}
#         mkdir -p ${root_dir}
#         python test.py \
#             --input_folder ${input_folder} \
#             --output_csv ${root_dir}/result.csv \
#             --output_folder ${root_dir} \
#             --enable_img_cond \
#             --cond_size ${res} \
#             --cfg ${cfg} > ${root_dir}/log.txt 2>&1
#     done
# done

# different condition image size with different input size 

input_size=768
cond_res=512
root_dir=results/diff_input_size_and_cond_size/kodak_input_${input_size}_img_cond_${cond_res}
mkdir -p ${root_dir}
python test.py \
    --input_folder ${input_folder} \
    --output_csv ${root_dir}/result.csv \
    --output_folder ${root_dir} \
    --input_size ${input_size} \
    --enable_img_cond \
    --cond_size ${cond_res} \
    --cfg 1.0 > ${root_dir}/log.txt 2>&1

input_size=512 
for cond_res in 64; do
    root_dir=results/diff_input_size_and_cond_size/kodak_input_${input_size}_img_cond_${cond_res}
    mkdir -p ${root_dir}
    python test.py \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/result.csv \
        --output_folder ${root_dir} \
        --input_size ${input_size} \
        --enable_img_cond \
        --cond_size ${cond_res} \
        --cfg 1.0 > ${root_dir}/log.txt 2>&1
done

input_size=384 
for cond_res in 64 128 256; do
    root_dir=results/diff_input_size_and_cond_size/kodak_input_${input_size}_img_cond_${cond_res}
    mkdir -p ${root_dir}
    python test.py \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/result.csv \
        --output_folder ${root_dir} \
        --input_size ${input_size} \
        --enable_img_cond \
        --cond_size ${cond_res} \
        --cfg 1.0 > ${root_dir}/log.txt 2>&1
done

input_size=256 
for cond_res in 64 128 ; do
    root_dir=results/diff_input_size_and_cond_size/kodak_input_${input_size}_img_cond_${cond_res}
    mkdir -p ${root_dir}
    python test.py \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/result.csv \
        --output_folder ${root_dir} \
        --input_size ${input_size} \
        --enable_img_cond \
        --cond_size ${cond_res} \
        --cfg 1.0 > ${root_dir}/log.txt 2>&1
done

input_size=128 
for cond_res in 64 ; do
    root_dir=results/diff_input_size_and_cond_size/kodak_input_${input_size}_img_cond_${cond_res}
    mkdir -p ${root_dir}
    python test.py \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/result.csv \
        --output_folder ${root_dir} \
        --input_size ${input_size} \
        --enable_img_cond \
        --cond_size ${cond_res} \
        --cfg 1.0 > ${root_dir}/log.txt 2>&1
done


# CLIC2020 test

mkdir -p results/clic2020
input_folder=/home/kris/generation_for_compression/dataset/CLIC2020_test_professional/


root_dir=results/clic2020/clic2020_normal
mkdir -p ${root_dir}
python test.py \
    --input_folder ${input_folder} \
    --output_csv ${root_dir}/result.csv \
    --output_folder ${root_dir} > ${root_dir}/log.txt 2>&1

root_dir=results/clic2020/different_input_size
for input_size in 1024 512 384 256 128 64; do
    mkdir -p ${root_dir}/size_${input_size}
    python test.py \
        --input_folder ${input_folder} \
        --output_csv ${root_dir}/size_${input_size}/result.csv \
        --output_folder ${root_dir}/size_${input_size} \
        --input_size ${input_size} > ${root_dir}/size_${input_size}/log.txt 2>&1
done