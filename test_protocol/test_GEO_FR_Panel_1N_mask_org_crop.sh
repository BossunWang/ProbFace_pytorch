CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.53 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.41 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.36 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.27 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.22 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.17 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log

CUDA_VISIBLE_DEVICES=1 python test_1N_type.py \
    --test_set 'GEO_FR_Panel_1N_mask_origin_crop' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --threshold -1.13 \
    --model_path "../prob_face_ir_101_masked_final" \
    2>&1 | tee -a log/GEO_FR_Panel_1N_mask_origin_crop.log


