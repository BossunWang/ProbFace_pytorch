mkdir 'log'
rm log/*.log
rm log/*.csv
rm -r 1N_features
sed -i 's/\r$//' test_GEO_crop_1N.sh
sh test_GEO_crop_1N.sh
sed -i 's/\r$//' test_GEO_env_crop_1N.sh
sh test_GEO_env_crop_1N.sh
sed -i 's/\r$//' test_GEO_env_mask_synthesis_crop_1N.sh
sh test_GEO_env_mask_synthesis_crop_1N.sh
sed -i 's/\r$//' test_GEO_identity_mask_synthesis_crop_1N.sh
sh test_GEO_identity_mask_synthesis_crop_1N.sh
sed -i 's/\r$//' test_GEO_Mask_Testing_Dataset_crop_1N.sh
sh test_GEO_Mask_Testing_Dataset_crop_1N.sh
sed -i 's/\r$//' test_GEO_FAR_crop_1N.sh
sh test_GEO_FAR_crop_1N.sh
sed -i 's/\r$//' test_GEO_FR_Panel_1N_org_crop.sh
sh test_GEO_FR_Panel_1N_org_crop.sh
sed -i 's/\r$//' test_MEDS_II_1N.sh
sh test_MEDS_II_1N.sh
