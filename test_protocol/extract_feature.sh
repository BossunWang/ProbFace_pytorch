CUDA_VISIBLE_DEVICES=1 python extract_feature.py \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 500 \
    --model_path '../test_models' \
    --feats_root 'feats' 

CUDA_VISIBLE_DEVICES=1 python extract_feature.py \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'iresnet100' \
    --backbone_conf_file '../training_mode/backbone_conf.yaml' \
    --batch_size 500 \
    --model_path '../test_models_iresnet' \
    --feats_root 'feats' 
