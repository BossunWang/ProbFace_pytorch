python remove_noises.py \
    --data_conf_file 'data_conf.yaml' \
    --remove_facescrub_noise 1 \
    --remove_megaface_noise 1 \
    --model_path '../test_models' \
    --facescrub_feature_dir 'feats/facescrub' \
    --facescrub_feature_outdir 'facescrub' \
    --megaface_feature_dir 'feats/megaface' \
    --megaface_feature_outdir 'megaface' \
    --masked_facescrub_feature_dir 'feats/masked_facescrub' \
    --masked_facescrub_feature_outdir 'masked_facescrub'

python remove_noises.py \
    --data_conf_file 'data_conf.yaml' \
    --remove_facescrub_noise 1 \
    --remove_megaface_noise 1 \
    --model_path '../test_models_iresnet' \
    --facescrub_feature_dir 'feats/facescrub' \
    --facescrub_feature_outdir 'facescrub' \
    --megaface_feature_dir 'feats/megaface' \
    --megaface_feature_outdir 'megaface' \
    --masked_facescrub_feature_dir 'feats/masked_facescrub' \
    --masked_facescrub_feature_outdir 'masked_facescrub'
