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
import dlib
import sys

abs_path = os.getcwd().split('ProbFace_pytorch')[0]
sys.path.append(os.path.join(abs_path, 'ProbFace_pytorch', "face_mask_adding/FMA-3D/"))
sys.path.append(os.path.join(abs_path, 'ProbFace_pytorch', "face_mask_adding/FMA-3D/utils"))
sys.path.append(os.path.join(abs_path, 'ProbFace_pytorch', "face_mask_adding/FMA-3D/models"))
sys.path.append(os.path.join(abs_path, 'ProbFace_pytorch', "face_mask_adding/FMA-3D/utils/cpython"))
from face_masker import FaceMasker


class ImageDataset(Dataset):
    def __init__(self, data_root, data_list_path
                 , img_size=112, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                 , sample_size=1, data_aug=False, data_aug_ratio=0.5, masked_ratio=0.5, device=None, prnet_model=None):
        self.data_root = data_root
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.sample_size = sample_size
        self.data_aug = data_aug
        self.data_aug_ratio = data_aug_ratio

        self.detector = dlib.get_frontal_face_detector()
        dlib_model_path = os.path.join(abs_path, 'ProbFace_pytorch'
                                       , "face_mask_adding/FMA-3D/shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(dlib_model_path)
        is_aug = True
        self.mask_offset_list = [0, 100]
        self.mask_template_number = 18
        prnet_model_path = os.path.join(abs_path, 'ProbFace_pytorch'
                                        , "face_mask_adding/FMA-3D/models/prnet.pth")
        index_path = os.path.join(abs_path, 'ProbFace_pytorch'
                                  , 'face_mask_adding/FMA-3D/Data/uv-data/face_ind.txt')
        triangles_path = os.path.join(abs_path, 'ProbFace_pytorch'
                                      , 'face_mask_adding/FMA-3D/Data/uv-data/triangles.txt')
        uv_face_path = os.path.join(abs_path, 'ProbFace_pytorch'
                                    , 'face_mask_adding/FMA-3D/Data/uv-data/uv_face_mask.png')
        mask_template_folder = os.path.join(abs_path, 'ProbFace_pytorch'
                                            , 'face_mask_adding/FMA-3D/Data/mask-data')
        self.face_masker_list \
            = [FaceMasker(is_aug, mask_offset
                          , prnet_model_path, index_path, triangles_path, uv_face_path, mask_template_folder
                          , device, prnet_model)
               for mask_offset in self.mask_offset_list]

        self.masked_ratio = masked_ratio

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

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def to_Tensor(self, image_path, face_masker=None, landmarks=None, template_name=None):
        img = cv2.imread(image_path)

        if face_masker is not None:
            img = face_masker.add_mask_from_img(img.copy(), landmarks, template_name, is_68_landmarks=True)

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.data_aug and random.random() > (1.0 - self.data_aug_ratio):
            blur_image_transform = self.blur_transform(image)
            return blur_image_transform
        else:
            image_transform = self.transform(image)
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

        for index, image_path in enumerate(file_list):
            image_path = os.path.join(self.data_root, cur_id, image_path)

            face_masker = None
            landmarks = None
            template_name = None

            if self.masked_ratio > 0. and index > self.sample_size // 2:
                img = dlib.load_rgb_image(image_path)
                dets = self.detector(img, 1)
                if len(dets) == 1 and random.random() > (1.0 - self.masked_ratio):
                    tmp_idx = np.random.randint(self.mask_template_number, size=1)[0]
                    mask_offset_index = np.random.randint(len(self.mask_offset_list), size=1)[0]
                    template_name = str(tmp_idx) + '.png'
                    face_masker = self.face_masker_list[mask_offset_index]

                    shape = self.predictor(img, dets[0])
                    landmarks = self.shape_to_np(shape, dtype='float')

            image = self.to_Tensor(image_path, face_masker, landmarks, template_name)
            image_list.append(image)

        image_list = torch.stack(image_list)

        return image_list, id_index


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import time
    import sys

    sys.path.append("../face_mask_adding/FMA-3D")
    from models.prnet import PRNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = '/media/glory/Transcend/Dataset/Face_Recognition_Dataset/WebFace260M/WebFace260M'
    data_list_path = "data.pkl"
    batch_size = 8

    prnet = PRNet(3, 3).to(device)
    prnet_model_path = "../face_mask_adding/FMA-3D/models/prnet.pth"
    state_dict = torch.load(prnet_model_path)
    prnet.load_state_dict(state_dict)
    prnet.eval()

    sample_size = 16
    data_aug = True
    id_loader = DataLoader(
        ImageDataset(data_root, data_list_path
                     , sample_size=sample_size, data_aug=data_aug, masked_ratio=1.0
                     , device=device, prnet_model=prnet),
        batch_size, shuffle=True, num_workers=0, drop_last=True)

    batch_iter = iter(id_loader)

    for batch_idx in range(10):
        data_time = time.time()
        images, labels = next(batch_iter)
        print(images.size())
        print(labels.size())
        print("data_time:", time.time() - data_time)
        for si, sample in enumerate(images[0]):
            plt.imsave('sample_images{}.png'.format(batch_idx * sample_size + si), (sample.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
