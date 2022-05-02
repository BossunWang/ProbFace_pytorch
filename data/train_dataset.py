"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, sampler
from torchvision import transforms
from PIL import Image
import pickle
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, data_root, data_list_path
                 , img_size=112, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                 , sample_size=1, data_aug=False):
        self.data_root = data_root
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.sample_size = sample_size
        self.data_aug = data_aug

        if os.path.isfile(data_list_path):
            with open(data_list_path, 'rb') as f:
                self.train_list = pickle.load(f)
        else:
            label_set = set()
            for dir, dirs, files in tqdm(os.walk(self.data_root)):
                if len(files) > 0:
                    id_name = dir.split('/')[-1]
                    label_set.add(id_name)

            self.train_list = list(label_set)
            with open(data_list_path, 'wb') as f:
                pickle.dump(self.train_list, f)

        print('Valid ids: %d.' % len(self.train_list))

        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.blur_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 11), sigma=(1, 2)),
            color_jitter,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.train_list)

    def to_Tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_transform = self.transform(image)
        if self.data_aug:
            blur_image_transform = self.blur_transform(image)
            return image_transform, blur_image_transform
        else:
            return image_transform

    def sample_list(self, file_list):
        if len(file_list) < self.sample_size:
            extend_file_list = []
            for i in range(self.sample_size // len(file_list)):
                extend_file_list.extend(file_list)
            mod_list = file_list[: self.sample_size % len(file_list)]
            extend_file_list.extend(mod_list)
        else:
            extend_file_list = file_list[:self.sample_size]

        extend_file_list = np.random.permutation(extend_file_list)
        return extend_file_list

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        id_index = np.array(index).repeat(self.sample_size)
        id_index = torch.from_numpy(id_index)
        file_list = os.listdir(os.path.join(self.data_root, cur_id))
        file_list = np.random.permutation(file_list)
        file_list = self.sample_list(file_list)

        image_list = []

        for image_path in file_list:
            image_path = os.path.join(self.data_root, cur_id, image_path)
            if self.data_aug:
                image, image_blur = self.to_Tensor(image_path)
                image_list.append(image)
                image_list.append(image_blur)
            else:
                image = self.to_Tensor(image_path)
                image_list.append(image)

        image_list = torch.stack(image_list)

        return image_list, id_index


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = '/media/glory/Transcend/Dataset/Face_Recognition_Dataset/WebFace260M/WebFace260M'
    data_list_path = "data.pkl"
    batch_size = 8

    id_loader = DataLoader(
        ImageDataset(data_root, data_list_path, sample_size=16, data_aug=False),
        batch_size, shuffle=True, num_workers=0, drop_last=True)

    batch_iter = iter(id_loader)

    for batch_idx in range(10):
        data_time = time.time()
        images, labels = next(batch_iter)
        sample = images[0][0]
        print(images.size())
        print(labels.size())
        print("data_time:", time.time() - data_time)
        plt.imsave('sample_images{}.png'.format(batch_idx), (sample.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

