python test_megaface.py \
    --data_conf_file 'data_conf.yaml' \
    --max_rank 10 \
    --model_path '../pretrained' \
    --facescrub_feature_dir '../../megaface_feats_cleaned/facescrub' \
    --megaface_feature_dir '../../megaface_feats_cleaned/megaface' \
    --masked_facescrub_feature_dir '../../megaface_feats_cleaned/masked_facescrub' \
    --is_concat 0 \
    2>&1 | tee -a log/megaface_log.log

python test_megaface.py \
    --data_conf_file 'data_conf.yaml' \
    --max_rank 10 \
    --model_path '../prob_face_ir_101_masked_final' \
    --facescrub_feature_dir '../../megaface_feats_cleaned/facescrub' \
    --megaface_feature_dir '../../megaface_feats_cleaned/megaface' \
    --masked_facescrub_feature_dir '../../megaface_feats_cleaned/masked_facescrub' \
    --is_concat 0 \
    2>&1 | tee -a log/megaface_log.log
