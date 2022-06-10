CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_joy_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.524 \
    --model_path '../test_models' \
    2>&1 | tee log/GEO_joy_crop_1N.log

CUDA_VISIBLE_DEVICES=4 python test_1N_type.py \
    --test_set 'GEO_joy_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold 0.524 \
    --model_path '../test_models' \
    2>&1 | tee -a log/GEO_joy_crop_1N.log
