CUDA_VISIBLE_DEVICES=4 python test_lfw_type.py \
    --test_set 'NIST_SD32_MEDS_II_face_FMA_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../test_models' \
    --offline_mode True \
    2>&1 | tee -a log/NIST_SD32_MEDS_II_face_FMA_mask.log

CUDA_VISIBLE_DEVICES=4 python test_lfw_type.py \
    --test_set 'NIST_SD32_MEDS_II_face_FMA_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'iresnet100' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../test_models_iresnet' \
    --offline_mode True \
    2>&1 | tee -a log/NIST_SD32_MEDS_II_face_FMA_mask.log

CUDA_VISIBLE_DEVICES=4 python test_lfw_type.py \
    --test_set 'NIST_SD32_MEDS_II_face_FMA_mask' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'iresnet100' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 100 \
    --model_path '../magface_origin_model' \
    --offline_mode True \
    --mean 0.0 \
    --std 255.0 \
    2>&1 | tee -a log/NIST_SD32_MEDS_II_face_FMA_mask.log
