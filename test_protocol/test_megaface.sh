python test_megaface.py \
    --data_conf_file 'data_conf.yaml' \
    --max_rank 10 \
    --model_path '../test_models' \
    --facescrub_feature_dir 'feats_cleaned/facescrub' \
    --megaface_feature_dir 'feats_cleaned/megaface' \
    --masked_facescrub_feature_dir 'feats_cleaned/masked_facescrub' \
    --is_concat 0 \
    2>&1 | tee -a log/megaface_log.log

python test_megaface.py \
    --data_conf_file 'data_conf.yaml' \
    --max_rank 10 \
    --model_path '../test_models_iresnet' \
    --facescrub_feature_dir 'feats_cleaned/facescrub' \
    --megaface_feature_dir 'feats_cleaned/megaface' \
    --masked_facescrub_feature_dir 'feats_cleaned/masked_facescrub' \
    --is_concat 0 \
    2>&1 | tee -a log/megaface_log.log
