python remove_noises.py \
    --data_conf_file 'data_conf.yaml' \
    --remove_facescrub_noise 1 \
    --remove_megaface_noise 1 \
    --model_path '../pretrained' \
    --facescrub_feature_dir '../../megaface_feats/facescrub' \
    --facescrub_feature_outdir 'facescrub' \
    --megaface_feature_dir '../../megaface_feats/megaface' \
    --megaface_feature_outdir 'megaface' \
    --masked_facescrub_feature_dir '../../megaface_feats/masked_facescrub' \
    --masked_facescrub_feature_outdir 'masked_facescrub' \
    --cleaned_feature_dir '../../megaface_feats_cleaned'

python remove_noises.py \
    --data_conf_file 'data_conf.yaml' \
    --remove_facescrub_noise 1 \
    --remove_megaface_noise 1 \
    --model_path '../prob_face_ir_101_masked_final' \
    --facescrub_feature_dir '../../megaface_feats/facescrub' \
    --facescrub_feature_outdir 'facescrub' \
    --megaface_feature_dir '../../megaface_feats/megaface' \
    --megaface_feature_outdir 'megaface' \
    --masked_facescrub_feature_dir '../../megaface_feats/masked_facescrub' \
    --masked_facescrub_feature_outdir 'masked_facescrub' \
    --cleaned_feature_dir '../../megaface_feats_cleaned'
