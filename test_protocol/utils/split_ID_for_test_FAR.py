import os
import numpy as np
from shutil import copyfile
import tqdm


if __name__ == '__main__':
    data_dir = '/workspace/data/face_dataset/1N_test_dataset/GEO_v1_march_mask_r480_c1831_ff_output_categorize/enroll'
    target_dir = '/workspace/data/face_dataset/1N_test_dataset/GEO_half_enroll'

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    id_set = set()
    for dir, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        for file in files:
            filepath = os.path.join(dir, file)

            if file.find('[N]') >= 0:
                id = file[file.find('[N]') + 3:file.find('[G]')]
                id_set.add(id)

    id_list = list(sorted(id_set))

    first_index = id_list.index('Abby')
    last_index = id_list.index('Joey')
    print('first index:', first_index)
    print('last index:', last_index)
    print('list size:', len(id_list))

    for dir, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        new_dir = dir.replace(data_dir, target_dir)

        for file in files:
            filepath = os.path.join(dir, file)

            if file.find('[N]') >= 0:
                id = file[file.find('[N]') + 3:file.find('[G]')]
                if id in id_list[first_index: last_index+1]:
                    new_filepath = os.path.join(new_dir, file)
                    copyfile(filepath, new_filepath)

