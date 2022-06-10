mkdir 'log'
rm log/*.*
#rm -r 11_features
sed -i 's/\r$//' test_lfw.sh
sh test_lfw.sh
sed -i 's/\r$//' test_lfw_mask0.sh
sh test_lfw_mask0.sh
sed -i 's/\r$//' test_rfw.sh
sh test_rfw.sh
sed -i 's/\r$//' test_rfw_mask0.sh
sh test_rfw_mask0.sh
sed -i 's/\r$//' test_calfw.sh
sh test_calfw.sh
sed -i 's/\r$//' test_calfw_mask0.sh
sh test_calfw_mask0.sh
sed -i 's/\r$//' test_cplfw.sh
sh test_cplfw.sh
sed -i 's/\r$//' test_cplfw_mask0.sh
sh test_cplfw_mask0.sh
sed -i 's/\r$//' test_cfp_fp.sh
sh test_cfp_fp.sh
sed -i 's/\r$//' test_cfp_fp_mask0.sh
sh test_cfp_fp_mask0.sh
sed -i 's/\r$//' test_agedb.sh
sh test_agedb.sh
sed -i 's/\r$//' test_agedb_mask0.sh
sh test_agedb_mask0.sh
sed -i 's/\r$//' test_mask_whn.sh
sh test_mask_whn.sh
sed -i 's/\r$//' test_NIST_SD32_MEDS_II.sh
sh test_MLFW.sh
sed -i 's/\r$//' test_MLFW.sh
sh test_MLFW.sh
#sh test_NIST_SD32_MEDS_II.sh
#sed -i 's/\r$//' test_NIST_SD32_MEDS_II_FMA_mask.sh
#sh test_NIST_SD32_MEDS_II_FMA_mask.sh


