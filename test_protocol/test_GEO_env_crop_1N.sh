CUDA_VISIBLE_DEVICES=5 python test_1N_type.py \
    --test_set 'GEO_env_crop_1N' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_env_crop_1N_log.log

CUDA_VISIBLE_DEVICES=5 python test_1N_type.py \
    --test_set 'GEO_env_crop_1N' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'iresnet100' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../test_models_iresnet' \
    2>&1 | tee -a log/GEO_env_crop_1N_log.log

CUDA_VISIBLE_DEVICES=5 python test_1N_type.py \
    --test_set 'GEO_env_crop_1N' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'iresnet100' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../magface_origin_model' \
    --mean 0.0 \
    --std 255.0 \
    2>&1 | tee -a log/GEO_env_crop_1N_log.log
