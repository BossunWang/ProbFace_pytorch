CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.102 \
    --model_path '../test_models' \
    2>&1 | tee log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.18 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.25 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.314 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.372 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.424 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log


CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.202 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.28 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.35 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.414 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.472 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_non_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.524 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_FR_Panel_1N_non_mask.log
