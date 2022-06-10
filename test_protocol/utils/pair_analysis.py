"""
@author: Jun Wang
@date: 20201016
@contact: jun21wangustc@gmail.com
"""

import sys
import numpy as np
from shutil import copyfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
# sys.path.append('/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/face_evaluation')
from extractor.feature_extractor import CommonExtractor


def sort_key(image_score):
    return image_score[-1]


def get_score_list(test_pair_list, image_name2feature):
    positive_score_list = []
    negtive_score_list = []
    for cur_pair in test_pair_list:
        image_name1 = cur_pair[0]
        feat1 = image_name2feature[image_name1]
        image_name2 = cur_pair[1]
        feat2 = image_name2feature[image_name2]
        label = cur_pair[2]
        cur_score = np.dot(feat1, feat2)
        cur_image_score = (image_name1, image_name2, cur_score)
        if label == 1:
            positive_score_list.append(cur_image_score)
        else:
            negtive_score_list.append(cur_image_score)
        positive_score_list.sort(key = sort_key)
        negtive_score_list.sort(key = sort_key)
    return positive_score_list, negtive_score_list


def parse_test_paris(pairs_file):
    test_pair_list = []
    pairs_file_buf = open(pairs_file)
    line1 = pairs_file_buf.readline().strip()
    while line1:
        line2 = pairs_file_buf.readline().strip()
        image_name1 = line1.split(' ')[0]
        image_name2 = line2.split(' ')[0]
        label = line1.split(' ')[1]
        test_pair_list.append((image_name1, image_name2, int(label)))
        line1 = pairs_file_buf.readline().strip()
    return test_pair_list


def save_score_list(score_file, score_list):
    score_file_buf = open(score_file, 'w')
    for image_score in score_list:
        image_name1 = image_score[0]
        image_name2 = image_score[1]
        score = image_score[2]
        score_file_buf.write(image_name1 + ' ' + image_name2 + ' ' + str(score) + '\n')


def eda_distribuion(feature_dict, masked_feature_dict, save_name, image_dir, mask_image_dir):
    outlier_dir = './outlier_images'
    if not os.path.isdir(outlier_dir):
        os.mkdir(outlier_dir)

    sigma_list = []
    for key, value in feature_dict.items():
        sigma_list.append(value[-1])

        if value[-1] > 1.:
            copyfile(os.path.join(image_dir, key), os.path.join(outlier_dir, key))

    masked_sigma_list = []
    for key, value in masked_feature_dict.items():
        masked_sigma_list.append(value[-1])

        if value[-1] > 1.:
            copyfile(os.path.join(mask_image_dir, key), os.path.join(outlier_dir, key))

    plt.figure(save_name)
    sns.distplot(sigma_list, label='non-maksed')
    sns.distplot(masked_sigma_list, label='masked')
    plt.legend()
    plt.savefig(save_name)


if __name__ == '__main__':
    # feats_root = '/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/face_evaluation/cplfw/cplfw_feats'
    # featureHandler = FeatureHandler(feats_root)
    # image_name2feature = featureHandler.load_feature()
    #
    # pairs_file = '/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/face_evaluation/cplfw/data/pairs_CPLFW.txt'
    # test_pair_list = parse_test_paris(pairs_file)
    #
    # positive_score_file = '/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/face_evaluation/cplfw/positive_score_list.txt'
    # negtive_score_file = '/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/face_evaluation/cplfw/negtive_score_list.txt'
    # positive_score_list, negtive_score_list = get_score_list(test_pair_list, image_name2feature)
    # save_score_list(positive_score_file, positive_score_list)
    # save_score_list(negtive_score_file, negtive_score_list)

    # analysis the distribution of MLS sigma scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = CommonExtractor(device)

    save_dir = '../MLS_Epoch_9_20210305.pt_LFW_features'
    imgae_dir = '/home/bossun/face_dataset/lfw_align_112/lfw/image'
    feature_dict = feature_extractor.load_feature(save_dir)
    print('lfw features size:', len(feature_dict))

    save_dir = '../MLS_Epoch_9_20210305.pt_LFW_MASK0_features'
    mask_image_dir = '/home/bossun/face_dataset/lfw_align_112/lfw/image_mask0'
    masked_feature_dict = feature_extractor.load_feature(save_dir)
    print('lfw masked features size:', len(masked_feature_dict))

    eda_distribuion(feature_dict, masked_feature_dict, 'LFW_MLS_sigma_score', imgae_dir, mask_image_dir)
