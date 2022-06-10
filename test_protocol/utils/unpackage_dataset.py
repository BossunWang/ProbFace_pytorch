import bcolz
import numpy as np
import os
import cv2
import tqdm


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def save_imgaes(carry, pair, img_path):
    for i, is_pair in tqdm.tqdm(enumerate(pair)):
        # print(carry[i * 2])
        image_p1 = np.array(carry[i * 2]).transpose((1, 2, 0))
        image_p2 = np.array(carry[i * 2 + 1]).transpose((1, 2, 0))

        image_p1 = (image_p1 * 0.5 + 0.5) * 255.0
        image_p2 = (image_p2 * 0.5 + 0.5) * 255.0
        image_p1 = image_p1.astype(np.uint8)
        image_p2 = image_p2.astype(np.uint8)

        cv2.imwrite(img_path + str(i) + '_p1.jpg', image_p1)
        cv2.imwrite(img_path + str(i) + '_p2.jpg', image_p2)


if __name__ == '__main__':
    calfw_path = '/home/bossun/face_dataset/calfw_align_112'
    file_name = 'calfw'
    carry, pair = get_val_pair(calfw_path, file_name)
    print(carry.shape)
    print(pair.shape)

    img_path = '/home/bossun/face_dataset/calfw_align_112/calfw/image/'
    save_imgaes(carry, pair, img_path)

    cplfw_path = '/home/bossun/face_dataset/cplfw_align_112'
    file_name = 'cplfw'
    carry, pair = get_val_pair(cplfw_path, file_name)
    print(carry.shape)
    print(pair.shape)

    img_path = '/home/bossun/face_dataset/cplfw_align_112/cplfw/image/'
    save_imgaes(carry, pair, img_path)
