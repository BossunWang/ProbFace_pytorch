"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import dlib
import cv2
from tqdm import tqdm
import numpy as np
import os
import time

from face_masker import FaceMasker
from utils.read_info import read_landmark_106_array


def unit_test():
    is_aug = True
    image_path = 'Data/test-data/test1.jpg'
    face_lms_file = 'Data/test-data/test1_landmark.txt'
    template_name = '17.png'
    masked_face_path = 'test1_half_mask1.png'
    face_lms_str = open(face_lms_file).readline().strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_lms = read_landmark_106_array(face_lms)

    # plt.plot(x, y, 'o', label='original data')
    # plt.plot(x, res.intercept + res.slope * x, 'r', label='fitted line')
    # plt.legend()
    # plt.show()

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for x, y in face_lms:
        x = int(x)
        y = int(y)
        cv2.circle(img_raw, (x, y), 3, (0, 255, 0), -1)

    cv2.imwrite('test1_half_mask1_landms.jpg', img_raw)

    mask_offset_list = [0, 100, 135, 155]

    for mask_offset in mask_offset_list:
        face_masker = FaceMasker(is_aug, mask_offset)
        masked_face_offset_path = masked_face_path.replace('.png', "_" + str(mask_offset) + '.png')
        # face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_offset_path, is_68_landmarks=True)
        image = cv2.imread(image_path)
        mask_image = face_masker.add_mask_from_img(image, face_lms, template_name, is_68_landmarks=True)
        cv2.imwrite(masked_face_offset_path, mask_image)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def mask_face_Dlib(detector
                   , predictor
                   , data_dir
                   , target_dir
                   , load_end_name='.jpg'
                   , save_landm_only=False):
    is_aug = True
    mask_offset_list = [0, 100, 135, 155]
    template_number = 18
    face_masker_list = [FaceMasker(is_aug, mask_offset) for mask_offset in mask_offset_list]

    for dir, dirs, files in tqdm(os.walk(data_dir)):
        new_dir = dir.replace(data_dir, target_dir)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        for file in files:
            filepath = os.path.join(dir, file)
            if not filepath.endswith(load_end_name):
                continue

            img = dlib.load_rgb_image(filepath)

            dets = detector(img, 1)
            if len(dets) == 0:
                continue

            detection_time = time.time()
            shape = predictor(img, dets[0])
            landmarks = shape_to_np(shape, dtype='float')
            print("detection_time:", time.time() - detection_time)

            if save_landm_only:
                target_image_path = filepath.replace(data_dir, target_dir)
                target_image_path = target_image_path.replace(".jpg", ".npy")
                np.save(target_image_path, landmarks)

                show_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                for x, y in landmarks:
                    x = int(x)
                    y = int(y)
                    cv2.circle(show_img, (x, y), 1, (0, 255, 0), -1)

                cv2.imshow("face landmark:", show_img)
                cv2.waitKey(1)
            else:
                img_raw = cv2.imread(filepath, cv2.IMREAD_COLOR)

                for tmp_idx in range(template_number):
                    template_name = str(tmp_idx) + '.png'
                    for inx, mask_offset in enumerate(mask_offset_list):
                        crop_image = img_raw.copy()
                        face_masker = face_masker_list[inx]
                        masked_time = time.time()
                        mask_image = face_masker.add_mask_from_img(crop_image, landmarks, template_name,
                                                                   is_68_landmarks=True)
                        print("masked_time:", time.time() - masked_time)
                        cv2.imshow("face masked:", mask_image)
                        cv2.waitKey(0)

                        target_image_path = filepath.replace(data_dir, target_dir)
                        target_image_path = target_image_path.replace(".jpg"
                                                                      , "_" + str(tmp_idx)
                                                                      + "_" + str(mask_offset) + ".png")
                        # print(target_image_path)
                        # print(target_folder)

                        cv2.imwrite(target_image_path, mask_image)


def main():
    # unit_test()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # data_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/imdb_retinafacecrop"
    # target_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/imdb_retinafacecrop_dlib_landmarks"
    # mask_face_Dlib(detector
    #                , predictor
    #                , data_dir
    #                , target_dir
    #                , save_landm_only=True)
    #
    # data_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/wiki_retinafacecrop"
    # target_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/wiki_retinafacecrop_dlib_landmarks"
    # mask_face_Dlib(detector
    #                , predictor
    #                , data_dir
    #                , target_dir
    #                , save_landm_only=True)

    data_dir = "/w/Bossun/face_dataset/UTKFace_crop"
    target_dir = "/w/Bossun/face_dataset/UTKFace_crop_dlib_landmarks"
    mask_face_Dlib(detector
                   , predictor
                   , data_dir
                   , target_dir
                   , save_landm_only=True)

    # data_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/imdb_retinafacecrop"
    # target_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/imdb_retinafacecrop_FMA_mask"
    # mask_face_Dlib(detector
    #                , predictor
    #                , data_dir
    #                , target_dir)
    #
    # data_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/wiki_retinafacecrop"
    # target_dir = "/media/glory/Transcend/Dataset/Face_Attribute_Dataset/Age_Gender/wiki_retinafacecrop_FMA_mask"
    # mask_face_Dlib(detector
    #                , predictor
    #                , data_dir
    #                , target_dir)


if __name__ == '__main__':
    main()
