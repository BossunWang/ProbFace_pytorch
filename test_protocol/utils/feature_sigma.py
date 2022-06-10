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


def calc_feature_sigma(data_dir, model, uncertainty_model, device):
    data_loader = DataLoader(TestDataset1N(data_dir), batch_size=1, num_workers=4, shuffle=True)

    sample_size = len(data_loader)

    feature_sigma = []
    for i, (image, image_path) in tqdm(enumerate(data_loader)):
        image = image.to(device)
        with torch.no_grad():
            mu, conv_final = model(image)
            logvar = uncertainty_model(conv_final)
            sigma_x = torch.exp(logvar).cpu().numpy()   
            feature_sigma.append(sigma_x)    

        if i > sample_size:
            break

    return np.array(feature_sigma)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_conf_file = '../../training_mode/backbone_conf.yaml'
    backbone_factory = BackboneFactory('ResnetUncertainty', backbone_conf_file)
    uncertainty_backbone_factory = BackboneFactory('UncertaintyHead', backbone_conf_file)
    
    model = backbone_factory.get_backbone()
    uncertainty_model = uncertainty_backbone_factory.get_backbone()
    
    model_path = '../../training_mode/MLS_training/eval_models/MLS_Epoch_9_mask_sst_magface.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['backbone'])
    uncertainty_dict = checkpoint['uncertainty_head']
    uncertainty_model.load_state_dict(uncertainty_dict)
    
    model = model.to(device)
    uncertainty_model = uncertainty_model.to(device)
    model.eval()
    uncertainty_model.eval()

    data_dirs = ['/workspace/data/face_dataset/lfw_align_112/lfw/image'
                 , '/workspace/data/face_dataset/lfw_align_112/lfw/image_mask0'
                 , '/workspace/data/face_dataset/cfp_align_112/cfp_align_112_extract_images/cfp_fp/image'
                 , '/workspace/data/face_dataset/cfp_align_112/cfp_align_112_extract_images/cfp_fp/image_mask0'
                 , '/workspace/data/face_dataset/MegaFace/face_crop_arcface/masked_facescrub_crop'
                 , '/workspace/data/face_dataset/1N_test_dataset/GEO_Mask_Testing_Dataset_crop_1N/identity']
    labels = ['lfw', 'lfw mask', 'cfp', 'cfp_mask', 'megaface mask', 'geo mask']

    for i, data_dir in enumerate(data_dirs):
        print(data_dir)
        feature_sigma = calc_feature_sigma(data_dir, model, uncertainty_model, device)
        sns.distplot(feature_sigma, hist=False, kde=True, label=labels[i])
    plt.legend()
    plt.savefig('sigma.jpg')
    # plt.show()


if __name__ == '__main__':
    main()