mkdir 'log'
rm log/*.log
rm log/*.csv
sed -i 's/\r$//' extract_feature.sh
sh extract_feature.sh
sed -i 's/\r$//' remove_noises.sh
sh remove_noises.sh
sed -i 's/\r$//' test_megaface.sh
sh test_megaface.sh

