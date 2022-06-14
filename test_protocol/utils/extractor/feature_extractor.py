"""
@author: Jun Wang
@date: 20201016 
@contact: jun21wangustc@gmail.com 
"""

import os
import logging as logger
import numpy as np
import torch
import tqdm
import torch.nn.functional as F

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class CommonExtractor:
    """Common feature extractor.
    
    Attributes:
        device(object): device to init model.
    """

    def __init__(self, device):
        self.device = torch.device(device)

    def extract_online(self, model, data_loader):
        """Extract and return features.
        
        Args:
            model(object): initialized model.
            data_loader(object): load data to be extracted.

        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        list_flag = ''
        if isinstance(model, list):   
            second_model = model[1] 
            second_model.eval()
            list_flag = model[2]  
            model = model[0]     

        model.eval()
        image_name2feature = {}
        with torch.no_grad():
            for batch_idx, (images, filenames) in tqdm.tqdm(enumerate(data_loader)):
                images = images.to(self.device)
                if list_flag == 'MLS':
                    mu, conv_final = model(images)
                    logvar = second_model(conv_final)
                    sigma_x = torch.exp(logvar)
                    features = torch.cat([mu, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'ProbFace':
                    feat, feature_fusions = model(images)
                    log_sigma_sq = second_model(feature_fusions)
                    sigma_x = torch.exp(log_sigma_sq)
                    feat = F.normalize(feat)
                    features = torch.cat([feat, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'SRT':
                    mu, conv_final = model(images)
                    output_feature = second_model(mu)
                    features = torch.cat([mu, output_feature], dim=1).cpu().numpy() 
                else:
                    feat, feature_fusions = model(images)
                    features = F.normalize(feat).cpu().numpy()
                for filename, feature in zip(filenames, features):
                    image_name2feature[filename] = feature

        return image_name2feature

    def extract_offline(self, feats_root, model, data_loader):
        """Extract and save features.

        Args:
            feats_root(str): the path to save features.
            model(object): initialized model.
            data_loader(object): load data to be extracted.
        """
        list_flag = ''
        if isinstance(model, list):   
            second_model = model[1] 
            second_model.eval()
            list_flag = model[2]  
            model = model[0]     

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, filenames) in tqdm.tqdm(enumerate(data_loader)):
                images = images.to(self.device)
                if list_flag == 'MLS':
                    mu, conv_final = model(images)
                    logvar = second_model(conv_final)
                    sigma_x = torch.exp(logvar)
                    features = torch.cat([mu, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'ProbFace':
                    feat, feature_fusions = model(images)
                    log_sigma_sq = second_model(feature_fusions)
                    sigma_x = torch.exp(log_sigma_sq)
                    feat = F.normalize(feat)
                    features = torch.cat([feat, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'SRT':
                    mu, conv_final = model(images)
                    output_feature = second_model(mu)
                    features = torch.cat([mu, output_feature], dim=1).cpu().numpy() 
                else:
                    feat, feature_fusions = model(images)
                    features = F.normalize(feat).cpu().numpy()

                for filename, feature in zip(filenames, features):
                    feature_name = os.path.splitext(filename)[0]
                    feature_path = os.path.join(feats_root, feature_name + '.npy')
                    feature_dir = os.path.dirname(feature_path)
                    if not os.path.exists(feature_dir):
                        os.makedirs(feature_dir)
                    np.save(feature_path, feature)
                if (batch_idx + 1) % 10 == 0:
                    logger.info('Finished batches: %d/%d.' % (batch_idx + 1, len(data_loader)))

    def extract_1N_offline(self, feats_root, model, enroll_data_loader, identity_data_loader):
        """Extract and save features.

        Args:
            feats_root(str): the path to save features.
            model(object): initialized model.
            data_loader(object): load data to be extracted.
        """
        list_flag = ''
        if isinstance(model, list):   
            second_model = model[1] 
            second_model.eval()
            list_flag = model[2]  
            model = model[0] 

        model.eval()
        with torch.no_grad():
            enroll_id_list = []
            enroll_feature_list = []
            identity_filename_list = []

            for batch_idx, (images, filenames) in tqdm.tqdm(enumerate(enroll_data_loader)):
                images = images.to(self.device)
                if list_flag == 'MLS':
                    mu, conv_final = model(images)
                    logvar = second_model(conv_final)
                    sigma_x = torch.exp(logvar)
                    features = torch.cat([mu, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'ProbFace':
                    feat, feature_fusions = model(images)
                    log_sigma_sq = second_model(feature_fusions)
                    sigma_x = torch.exp(log_sigma_sq)
                    feat = F.normalize(feat)
                    features = torch.cat([feat, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'SRT':
                    mu, conv_final = model(images)
                    output_feature = second_model(mu)
                    features = torch.cat([mu, output_feature], dim=1).cpu().numpy() 
                else:
                    feat, feature_fusions = model(images)
                    features = F.normalize(feat).cpu().numpy()

                for filename, feature in zip(filenames, features):
                    if filename.find('GEO') >= 0:
                        if filename.find('[N]') >= 0:
                            id = filename[filename.find('[N]') + 3:filename.find('[G]')]
                            id = id.replace('.', ' ')
                        else:
                            file_name = filename.split('/')[-1]
                            id = file_name[0:file_name.find('_2')]
                    else:
                        id = filename.split('/')[-1].split('-')[0]

                    enroll_id_list.append(id)
                    enroll_feature_list.append(feature.reshape(-1))

            identity_id_list = [None] * len(enroll_id_list)
            identity_feature_list = []
            global_index = 0
            for batch_idx, (images, filenames) in tqdm.tqdm(enumerate(identity_data_loader)):
                images = images.to(self.device)
                if list_flag == 'MLS':
                    mu, conv_final = model(images)
                    logvar = second_model(conv_final)
                    sigma_x = torch.exp(logvar)
                    features = torch.cat([mu, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'ProbFace':
                    feat, feature_fusions = model(images)
                    log_sigma_sq = second_model(feature_fusions)
                    sigma_x = torch.exp(log_sigma_sq)
                    feat = F.normalize(feat)
                    features = torch.cat([feat, sigma_x], dim=1).cpu().numpy()
                elif list_flag == 'SRT':
                    mu, conv_final = model(images)
                    output_feature = second_model(mu)
                    features = torch.cat([mu, output_feature], dim=1).cpu().numpy() 
                else:
                    features = F.normalize(model(images)).cpu().numpy()

                for i, (filename, feature) in enumerate(zip(filenames, features)):
                    if filename.find('GEO') >= 0:
                        if filename.find('[N]') >= 0:
                            id = filename[filename.find('[N]') + 3:filename.find('[G]')]
                        elif len(filename.split('/')[-1].split('_')) >= 2:
                            file_name = filename.split('/')[-1]
                            id = file_name.split('_')[0]                            
                        else:
                            file_name = filename.split('/')[-1]
                            id = file_name[0:file_name.find('_2')]
                        
                        id = id.replace('.', ' ')
                    else:
                        id = filename.split('/')[-1].split('-')[0]

                    for enroll_index, enroll_id in enumerate(enroll_id_list):
                        if identity_id_list[enroll_index] is None:
                            identity_id_list[enroll_index] = []
                        if id == enroll_id:
                            identity_id_list[enroll_index].append(global_index + i)

                    identity_filename_list.append(filename)
                    identity_feature_list.append(feature.reshape(-1))
                global_index += images.size(0)

        if not os.path.exists(feats_root):
            os.makedirs(feats_root)

        np.save(feats_root + 'enroll_id_list', enroll_id_list)
        np.save(feats_root + 'identity_filename_list', identity_filename_list)
        np.save(feats_root + 'enroll_feature_list', enroll_feature_list)
        np.save(feats_root + 'identity_id_list', identity_id_list)
        np.save(feats_root + 'identity_feature_list', identity_feature_list)

    def extract_1N_from_jason(self, feats_root, enroll_data_loader, identity_data_loader):
        enroll_id_list = []
        enroll_feature_list = []
        identity_filename_list = []

        for batch_idx, (features, filenames) in tqdm.tqdm(enumerate(enroll_data_loader)):
            features = features.to(self.device)
            features = features.cpu().numpy()

            for filename, feature in zip(filenames, features):
                id = filename[filename.find('[N]') + 3:filename.find('[G]')]
                id = id.replace('.', ' ')

                enroll_id_list.append(id)
                enroll_feature_list.append(feature.reshape(-1))

        identity_id_list = [None] * len(enroll_id_list)
        identity_feature_list = []
        global_index = 0

        for batch_idx, (features, filenames) in tqdm.tqdm(enumerate(identity_data_loader)):
            features = features.to(self.device)
            features = features.cpu().numpy()

            for i, (filename, feature) in enumerate(zip(filenames, features)):
                id = filename.split('_')[0]
                id = id.replace('.', ' ')

                for enroll_index, enroll_id in enumerate(enroll_id_list):
                    if identity_id_list[enroll_index] is None:
                        identity_id_list[enroll_index] = []
                    if id == enroll_id:
                        identity_id_list[enroll_index].append(global_index + i)

                identity_filename_list.append(filename)
                identity_feature_list.append(feature.reshape(-1))
            global_index += features.shape[0]

        if not os.path.exists(feats_root):
            os.makedirs(feats_root)

        np.save(feats_root + 'enroll_id_list', enroll_id_list)
        np.save(feats_root + 'identity_filename_list', identity_filename_list)
        np.save(feats_root + 'enroll_feature_list', enroll_feature_list)
        np.save(feats_root + 'identity_id_list', identity_id_list)
        np.save(feats_root + 'identity_feature_list', identity_feature_list)

    def load_feature(self, feats_root):
        """Load features to memory.

        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        image_name2feature = {}
        for root, dirs, files in os.walk(feats_root):
            for cur_file in files:
                if cur_file.endswith('.npy'):
                    cur_file_path = os.path.join(root, cur_file)
                    cur_feats = np.load(cur_file_path)
                    if feats_root.endswith('/'):
                        cur_short_path = cur_file_path[len(feats_root):]
                    else:
                        cur_short_path = cur_file_path[len(feats_root) + 1:]
                    cur_key = cur_short_path.replace('.npy', '.jpg')
                    image_name2feature[cur_key] = cur_feats
        return image_name2feature

    def load_1N_feature(self, feats_root):
        enroll_feature_list = np.load(feats_root + 'enroll_feature_list.npy')
        identity_feature_list = np.load(feats_root + 'identity_feature_list.npy')
        enroll_id_list = np.load(feats_root + 'enroll_id_list.npy', allow_pickle=True)
        identity_filename_list = np.load(feats_root + 'identity_filename_list.npy', allow_pickle=True)
        identity_id_list = np.load(feats_root + 'identity_id_list.npy', allow_pickle=True)

        return enroll_feature_list \
            , identity_feature_list \
            , enroll_id_list \
            , identity_filename_list \
            , identity_id_list
