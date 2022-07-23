CUDA_VISIBLE_DEVICES=1 python extract_feature.py \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --batch_size 100 \
    --model_path '../pretrained' \
    --load_backbone_only \
    --feats_root '../../megaface_feats'

CUDA_VISIBLE_DEVICES=1 python extract_feature.py \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ir_101' \
    --batch_size 100 \
    --model_path '../prob_face_ir_101_masked_final' \
    --feats_root '../../megaface_feats'
