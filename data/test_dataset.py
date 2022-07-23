"""
@author: Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import logging as logger
import cv2
import numpy as np
import natsort
from PIL import Image
import torch
from torch.utils.data import Dataset
import json


logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_p1_root, image_p2_root, pair_list, crop_eye=False, mean=127.5, std=128.0):
        # for RFW mask
        self.image_p1_root = image_p1_root
        self.image_p2_root = image_p2_root
        dirFiles = []
        for pair_line in pair_list:
            dirFiles.append(pair_line[0])
            dirFiles.append(pair_line[1])
        self.image_list = dirFiles
        self.crop_eye = crop_eye
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if index % 2 == 0:
            image_root = self.image_p1_root
        else:
            image_root = self.image_p2_root
        short_image_path = self.image_list[index]
        image_path = os.path.join(image_root, short_image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = cv2.resize(image, (128, 128))
        if self.crop_eye:
            image = image[:60, :]
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, short_image_path


class TestDataset1N(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, crop_eye=False, mean=127.5, std=128.0):
        self.image_root = image_root
        self.image_list = []

        for dir, dirs, files in os.walk(image_root):
            for file in files:
                self.image_list.append(os.path.join(dir, file))

        self.crop_eye = crop_eye
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_eye:
            image = image[:60, :]
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, image_path


class TestDataset1NFromJason(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        json_root(str): root directory of test set.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, json_root, crop_eye=False):
        self.json_root = json_root
        self.json_list = []

        for dir, dirs, files in os.walk(json_root):
            for file in files:
                self.json_list.append(os.path.join(dir, file))

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        file = open(self.json_list[index])
        data = json.load(file)
        features = torch.from_numpy(np.array(data['features'], dtype=np.float32))
        # torch.set_printoptions(precision=8)
        # print('data:', data['features'][0])
        # print('numpy:', np.array(data['features'], dtype=np.float32)[0])
        # print('feature:', features[0])
        return features, data['org_file_name']


class TestDatasetMegaFace(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, crop_eye=False, mean=127.5, std=128.0):
        self.image_root = image_root
        self.image_list = []

        for dir, dirs, files in os.walk(image_root):
            for file in files:
                self.image_list.append(os.path.join(dir, file))

        self.crop_eye = crop_eye
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_eye:
            image = image[:60, :]
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, image_path.replace(self.image_root + '/', '')
