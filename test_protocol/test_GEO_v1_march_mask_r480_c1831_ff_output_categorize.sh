CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResnetUncertainty' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../training_mode/MLS_training/eval_models' \
    2>&1 | tee log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.102 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.18 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.25 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.314 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.372 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.424 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log


CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.202 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.28 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.35 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.414 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.472 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_v1_march_mask_r480_c1831_ff_output_categorize' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.524 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_v1_march_mask_r480_c1831_ff_output_categorize.log
