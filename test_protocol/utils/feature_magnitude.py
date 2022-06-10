import sys
sys.path.append('../../')
from backbone.backbone_def import BackboneFactory
from data_processor.test_dataset import TestDataset1N
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def calc_feature_magnitude(data_dir, model, device):
    data_loader = DataLoader(TestDataset1N(data_dir), batch_size=1, num_workers=4, shuffle=True)

    sample_size = len(data_loader)

    feature_magnitude = []
    for i, (image, image_path) in tqdm(enumerate(data_loader)):
        image = image.to(device)
        with torch.no_grad():
            feature = model(image)
            feature = feature.squeeze(0).detach().cpu().numpy()
            magnitude = np.linalg.norm(feature)
            feature_magnitude.append(magnitude)

        if i > sample_size:
            break

    return np.array(feature_magnitude)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_factory = BackboneFactory('ResNet', '../../training_mode/backbone_conf.yaml')
    model = backbone_factory.get_backbone()
    model_path = '../../test_models/SST_Prototype_Epoch_29_nonmask_mask_20210730.pt'
    model.load_state_dict(torch.load(model_path)['backbone'])
    model = model.to(device)
    model.eval()

    data_dirs = ['/workspace/data/face_dataset/lfw_align_112/lfw/image'
                 , '/workspace/data/face_dataset/lfw_align_112/lfw/image_mask0'
                 , '/workspace/data/face_dataset/cfp_align_112/cfp_align_112_extract_images/cfp_fp/image'
                 , '/workspace/data/face_dataset/cfp_align_112/cfp_align_112_extract_images/cfp_fp/image_mask0'
                 , '/workspace/data/face_dataset/MegaFace/face_crop_arcface/masked_facescrub_crop'
                 , '/workspace/data/face_dataset/1N_test_dataset/GEO_Mask_Testing_Dataset_crop_1N/identity']
    labels = ['lfw', 'lfw mask', 'cfp', 'cfp_mask', 'megaface mask', 'geo mask']

    for i, data_dir in enumerate(data_dirs):
        print(data_dir)
        feature_magnitude = calc_feature_magnitude(data_dir, model, device)
        sns.distplot(feature_magnitude, hist=False, kde=True, label=labels[i])
    plt.legend()
    plt.savefig('magnitude.jpg')
    # plt.show()


if __name__ == '__main__':
    main()