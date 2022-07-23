""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""  

import sys
import yaml
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
abs_path = os.getcwd().split('ProbFace_pytorch')[0]
sys.path.append(os.path.join(abs_path, 'ProbFace_pytorch'))
from data.test_dataset import TestDatasetMegaFace

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--data_conf_file", type=str,
                      help="The path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type=str,
                      help="Resnet, Mobilefacenets.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help="The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type=int, default=1024)
    conf.add_argument('--model_path', type=str, default='mv_epoch_8.pt',
                      help='The path of model')
    conf.add_argument('--feats_root', type=str, default='mv_epoch_8.pt',
                      help='The path for feature save.')
    conf.add_argument('--mean', type=float, default=127.5)
    conf.add_argument('--std', type=float, default=127.5)
    conf.add_argument("--load_backbone_only", action='store_true', default=False)
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)['MegaFace']
        croped_facescrub_folder = data_conf['croped_facescrub_folder']        
        croped_megaface_folder = data_conf['croped_megaface_folder']        
        megaface_mask = data_conf['megaface-mask']
        masked_croped_facescrub_folder = data_conf['masked_croped_facescrub_folder']
    
    facescrub_data_loader = DataLoader(TestDatasetMegaFace(croped_facescrub_folder, False, args.mean, args.std), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    megaface_data_loader = DataLoader(TestDatasetMegaFace(croped_megaface_folder, False, args.mean, args.std), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # define model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(args.backbone_type, device)

    feature_extractor = CommonExtractor(device)

    print('facescrub dataset size:', len(facescrub_data_loader.dataset))
    print('megaface dataset size:', len(megaface_data_loader.dataset))
    
    if os.path.isdir(args.model_path):
        model_name_list = os.listdir(args.model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                print('test %s' % model_name)
                model_path = os.path.join(args.model_path, model_name)
                model = model_loader.load_model(model_path, args.load_backbone_only)

                method_name = os.path.basename(model_path)
                facescrub_save_dir = os.path.join(args.feats_root, 'facescrub', model_name)
                megaface_save_dir = os.path.join(args.feats_root, 'megaface', model_name)
    
                # extract feature.                
                feature_extractor.extract_offline(facescrub_save_dir, model, facescrub_data_loader)
                feature_extractor.extract_offline(megaface_save_dir, model, megaface_data_loader)
                
                if megaface_mask == 1:
                    data_loader = DataLoader(TestDatasetMegaFace(masked_croped_facescrub_folder, False, args.mean, args.std), 
                                            batch_size=args.batch_size, num_workers=4, shuffle=False)
                    masked_facescrub_save_dir = os.path.join(args.feats_root, 'masked_facescrub', model_name)
                    feature_extractor.extract_offline(masked_facescrub_save_dir, model, data_loader)
    else:
        print('model path should be the type of dir.')
