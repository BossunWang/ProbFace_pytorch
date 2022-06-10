CUDA_VISIBLE_DEVICES=1 python test_lfw_type.py \
    --test_set 'Masked_whn' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --batch_size 100 \
    --model_path '../pretrained' \
    --offline_mode True \
    --load_backbone_only \
    2>&1 | tee -a log/masked_whn_log.log

CUDA_VISIBLE_DEVICES=1 python test_lfw_type.py \
    --test_set 'Masked_whn' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --batch_size 100 \
    --model_path '../prob_face_ir_101_masked_final' \
    --offline_mode True \
    2>&1 | tee -a log/masked_whn_log.log
