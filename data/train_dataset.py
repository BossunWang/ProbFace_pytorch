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
                self.train_list  = pickle.load(f)
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
        blur_image_transform = self.blur_transform(image)

        return image_transform, blur_image_transform

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
        file_list = os.listdir(os.path.join(self.data_root, cur_id))
        file_list = np.random.permutation(file_list)
        file_list = self.sample_list(file_list)

        image_list = []

        for image_path in file_list:
            image_path = os.path.join(self.data_root, cur_id, image_path)

            image, image_blur = self.to_Tensor(image_path)

            image_list.append(image)
            if self.data_aug:
                image_list.append(image_blur)

        image_list = torch.stack(image_list)

        return image_list, cur_id


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = '/media/glory/Transcend/Dataset/Face_Recognition_Dataset/WebFace260M/WebFace260M'
    data_list_path = "data.pkl"
    batch_size = 8

    id_loader = DataLoader(
        ImageDataset(data_root, data_list_path, sample_size=16),
        batch_size, shuffle=True, num_workers=0, drop_last=True)

    for batch_idx, (images, labels) in enumerate(id_loader):
        sample = images[0][0]
        print(sample.size())
        plt.imsave('sample_images{}.png'.format(batch_idx), (sample.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
        if batch_idx >= 10:
            break

    # exclude_id_set = set()
    # data_loader = DataLoader(
    #     ImageTripletDataset(easy_data_root, hard_data_root),
    #     batch_size, True, num_workers=1, drop_last=True)
    #
    # for batch_idx, (easy_pos_images, hard_anchor_images, easy_neg_images, image_labels) in enumerate(data_loader):
    #     print(easy_pos_images.size())
    #     print(hard_anchor_images.size())
    #     print(easy_neg_images.size())
    #     print(image_labels)
    #
    #     easy_pos_images = easy_pos_images.reshape(-1, easy_pos_images.size(2), easy_pos_images.size(3), easy_pos_images.size(4))
    #     hard_anchor_images = hard_anchor_images.reshape(-1, hard_anchor_images.size(2), hard_anchor_images.size(3), hard_anchor_images.size(4))
    #     easy_neg_images = easy_neg_images.reshape(-1, easy_neg_images.size(2), easy_neg_images.size(3), easy_neg_images.size(4))
    #
    #     pos_result = torch.cat((easy_pos_images[0], hard_anchor_images[0]), 2)
    #     plt.imsave('pos_sample.png', (pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #
    #     neg_result = torch.cat((easy_neg_images[0], hard_anchor_images[0]), 2)
    #     plt.imsave('neg_sample.png', (neg_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    # exclude_id_set = set()
    # sample_size = 4
    # data_loader = DataLoader(
    #     ImageDataset_SST(easy_data_root, hard_data_root, exclude_id_set, sample_size=sample_size, data_aug=True),
    #     batch_size, True, num_workers=0, drop_last=True)
    #
    # for batch_idx, (images1, images2, labels) in tqdm(enumerate(data_loader)):
    #     _, _, c, w, h = images1.size()
    #     images1 = images1.view(-1, c, w, h)
    #     images2 = images2.view(-1, c, w, h)
    #
    #     # print(labels)
    #     # print(images1[::sample_size].size())
    #
    #     sample = torch.cat((images1[0], images2[0]), 2)
    #     sample_blur = torch.cat((images1[sample_size], images2[sample_size]), 2)
    #     plt.imsave('sample.png', (sample.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     plt.imsave('sample_blur.png', (sample_blur.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    # rand_indexs = np.random.permutation(data_loader.dataset.train_list)

    # print(rand_indexs)

    # train_dataset = ImageEasyAndHardDataset(easy_data_root, hard_data_root)
    #
    # print(len(train_dataset))
    #
    # data_loader = DataLoader(train_dataset,
    #                         batch_size=batch_size,
    #                         num_workers=4, shuffle=True)
    #
    # for batch_idx, (images, labels) in enumerate(data_loader):
    #     images = images.reshape(-1, images.size(2), images.size(3), images.size(4))
    #     labels = labels.reshape(-1)
    #     print(images.size())
    #     print(labels.size())
    #
    #     pos_result = torch.cat((images[0], images[1]), 2)
    #     plt.imsave('pos_sample.png', (pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    # from torch.utils.data import DataLoader
    # # test iter data_loader
    # train_dataset = ImageDataset('/workspace/data/public/FR/MS-Celeb-1M/V3/MS1M-V3'
    #                             , '../utils/ms1m_v3_file.txt')
    #
    # data_loader = iter(DataLoader(
    #                     train_dataset,
    #                     batch_size=16,
    #                     sampler=InfiniteSamplerWrapper(train_dataset),
    #                     num_workers=4))
    #
    # images, labels = next(data_loader)
    #
    # print(images.size())
    # print(labels.size())
    #
    # import sys
    # import matplotlib.pyplot as plt
    # import tqdm
    
    # sys.path.append('../backbone/')
    # from ResNets import Resnet
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Resnet(100, 0.4, mode='ir_se', feat_dim=512, out_h=7, out_w=7).to(device)
    
    # state_dict = torch.load('../pretrain_model/Backbone_IR_SE_101_Epoch_24_Time_2020-10-26-01-56_checkpoint.pth'
    #                         , map_location=device)
    # model.load_state_dict(state_dict)
    
    # model = torch.nn.DataParallel(model)
    # model.eval()
    
    # print('load model finished')
    
    # easy_data_root = '/workspace/data/public/FR/MS-Celeb-1M/V3/MS1M-V3'
    # # easy_data_root = '/home/bossun/face_dataset/ms1m_align_112'
    # # should be set mask dataset
    # hard_data_root = '/workspace/data/face_dataset/MS1M-V3_FMA_mask'
    # # hard_data_root = '/home/bossun/face_dataset/ms1m_align_112'
    # easy_positive_pairs = '../utils/easy_positive_pairs.txt'
    # hard_positive_pairs = '../utils/hard_positive_pairs.txt'
    # dataset = ImageDataset_Pos_Neg(easy_data_root, hard_data_root, easy_positive_pairs, hard_positive_pairs, device)
    
    # # simulate one epoch
    # batch_size = 16
    # rand_indexs = np.random.permutation(dataset.len())
    
    # for start_index in tqdm.tqdm(range(0, dataset.len(), batch_size)):
    #     end_index = start_index + batch_size if start_index + batch_size <= dataset.len() else dataset.len()
    #     # print(len(rand_indexs[start_index:end_index]))
    
    #     easy_pos_samples1, easy_pos_samples2 \
    #         , hard_pos_samples1, hard_pos_samples2 \
    #         , easy_neg_samples1, easy_neg_samples2 \
    #         , hard_neg_samples1, hard_neg_samples2 \
    #         , easy_positive_image_labels1, easy_positive_image_labels2 \
    #         , hard_positive_image_labels1, hard_positive_image_labels2 \
    #         , easy_negative_image_labels1, easy_negative_image_labels2 \
    #         , hard_negative_image_labels1, hard_negative_image_labels2 \
    #         = dataset.sample(rand_indexs[start_index:end_index], model)
    
    #     # print(easy_pos_samples1.size())
    #     # print(easy_positive_image_labels1.size())
    
    #     easy_pos_result = torch.cat((easy_pos_samples1[0], easy_pos_samples2[0]), 2)
    #     hard_pos_result = torch.cat((hard_pos_samples1[0], hard_pos_samples2[0]), 2)
    #     easy_neg_result = torch.cat((easy_neg_samples1[0], easy_neg_samples2[0]), 2)
    #     hard_neg_result = torch.cat((hard_neg_samples1[0], hard_neg_samples2[0]), 2)
    
    #     plt.imsave('easy_pos_sample.png', (easy_pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     plt.imsave('hard_pos_sample.png', (hard_pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     plt.imsave('easy_neg_sample.png', (easy_neg_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     plt.imsave('hard_neg_sample.png', (hard_neg_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)