import os
import numpy as np
from shutil import copyfile
import tqdm


def split_dataset(data_dir, mask_target_dir, non_mask_target_dir):
    if not os.path.isdir(mask_target_dir):
        os.makedirs(mask_target_dir)

    if not os.path.isdir(non_mask_target_dir):
        os.makedirs(non_mask_target_dir)

    for dir, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        for file in files:
            if file.endswith('.txt'):
                continue

            filepath = os.path.join(dir, file)
            new_file_name = '_'.join(file.split('_')[5:])

            if file.split('_')[1] == '0':
                new_file_path = os.path.join(non_mask_target_dir, new_file_name)
            else:
                new_file_path = os.path.join(mask_target_dir, new_file_name)

            copyfile(filepath, new_file_path)


if __name__ == '__main__':
    data_dir = '/home/bossun/face_dataset/gv_040_test_output_0504y21_pretest_2173_gv040_output_json/'
    mask_target_dir = '/home/bossun/face_dataset/gv_040_test_output_0504y21_pretest_2173_gv040_output_json_mask'
    non_mask_target_dir = '/home/bossun/face_dataset/gv_040_test_output_0504y21_pretest_2173_gv040_output_json_non_mask'

    split_dataset(data_dir, mask_target_dir, non_mask_target_dir)

    data_dir = '/home/bossun/face_dataset/gv_040_test_output_0504y21_pretest_2173_gv040_output_json_origin_crop/'
    mask_target_dir = '/home/bossun/face_dataset/1N_test_dataset/GEO_FR_Panel_1N_mask_origin_crop/identity'
    non_mask_target_dir = '/home/bossun/face_dataset/1N_test_dataset/GEO_FR_Panel_1N_non_mask_origin_crop/identity'

    split_dataset(data_dir, mask_target_dir, non_mask_target_dir)