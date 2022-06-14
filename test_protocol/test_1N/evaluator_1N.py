"""
@author: Bossun Wang
@date: 20210309
@contact: vvmodouco@gmail.com
"""

import os
import sys
import numpy as np
from shutil import copyfile
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append('../')
from utils.pair_metric import MLS_score_matrix


class Evaluator1N(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, enroll_data_loader, identity_data_loader, feature_extractor):
        """Init LFWEvaluator.

        Args:
            enroll_data_loader(object): a test data loader for enroll images.
            identity_data_loader(object): a test data loader for identity images.
            feature_extractor(object): a feature extractor.
        """
        self.enroll_data_loader = enroll_data_loader
        self.identity_data_loader = identity_data_loader
        self.feature_extractor = feature_extractor

    def pair_cosin_score(self, x1, x2):
        cur_score = np.dot(x1, x2.T)
        return cur_score

    def test(self, model, threshold, save_dir=None, only_load=False, from_json=False):
        if from_json:
            self.feature_extractor.extract_1N_from_jason(save_dir + '_features/'
                                                         , self.enroll_data_loader
                                                         , self.identity_data_loader)
            enroll_feature_list \
                , identity_feature_list \
                , enroll_id_list \
                , identity_filename_list \
                , identity_id_list = self.feature_extractor.load_1N_feature(save_dir + '_features/')

        elif only_load:
            enroll_feature_list \
                , identity_feature_list \
                , enroll_id_list \
                , identity_filename_list \
                , identity_id_list = self.feature_extractor.load_1N_feature(save_dir + '_features/')
        else:
            self.feature_extractor.extract_1N_offline(save_dir + '_features/'
                                                      , model
                                                      , self.enroll_data_loader
                                                      , self.identity_data_loader)
            enroll_feature_list \
                , identity_feature_list \
                , enroll_id_list \
                , identity_filename_list \
                , identity_id_list = self.feature_extractor.load_1N_feature(save_dir + '_features/')
        
        top1_tar, top1_far, top1_frr, top1_tnr, confusion_type_name_list, analysis_index_list, analysis_gallery_index_list, score_list, G_score_list, I_score_list, FDR, positive_score_mean, negative_score_mean \
                        = self.test_one_model(identity_feature_list, enroll_feature_list, enroll_id_list, identity_id_list, threshold)

        stored_img_dir = save_dir + '_analysis_images/'
        if not os.path.exists(stored_img_dir):
            os.mkdir(stored_img_dir)

        for i, analysis_index in enumerate(analysis_index_list):
            filename = identity_filename_list[analysis_index]

            confuntion_type = confusion_type_name_list[i]
            target_dir = os.path.join(stored_img_dir, confuntion_type)

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            target_filename = os.path.join(target_dir, filename.split('/')[-1])
            target_filename = target_filename \
                              + '_t1_' + enroll_id_list[analysis_gallery_index_list[i * 2]] \
                              + '_' + str("%.2f" % score_list[i * 2]) \
                              + '_t2_' + enroll_id_list[analysis_gallery_index_list[i * 2 + 1]] \
                              + '_' + str("%.2f" % score_list[i * 2 + 1]) + '.jpg'
            # print(filename, target_filename)
            if not from_json:
                copyfile(filename, target_filename)

        plt.figure('GI_dist')
        G_score_array = np.array(G_score_list)
        I_score_array = np.concatenate((np.array(I_score_list), np.array(score_list[1::2])), axis=0)
        sns.distplot(G_score_array, hist=False, kde=True, label='G_score')
        sns.distplot(I_score_array, hist=False, kde=True, label='I_score')
        plt.legend()
        plt.savefig(stored_img_dir + 'GI_dist.jpg')
        plt.close('all')

        return top1_tar, top1_far, top1_frr, top1_tnr, FDR, positive_score_mean, negative_score_mean

    def test_one_model(self, query_feats, gallery_feats, enroll_id_list, identity_id_list, threshold): 
        query_num = query_feats.shape[0]
        gallery_num = gallery_feats.shape[0]

        MLS_flag = False

        if query_feats.shape[-1] == 257 or query_feats.shape[-1] == 513:
            # cos_similarity = self.pair_cosin_score(query_feats[:, :-1], gallery_feats[:, :-1])
            # top_inds = np.argsort(-cos_similarity, axis=1)
            #
            # # selcet candidate for calculate MLS
            # candidate = 30
            # similarity = np.ones((query_num, gallery_num)) * -100.0
            # for qi, query_feat in enumerate(query_feats):
            #     j = top_inds[qi, 0:candidate]
            #     for index_j in j:
            #         similarity[qi][index_j] = self.pair_MLS_score(query_feat, gallery_feats[index_j])
            #
            # MLS_flag = True
            similarity = MLS_score_matrix(query_feats[:, :512], gallery_feats[:, :512]
                                          , query_feats[:, -1], gallery_feats[:, -1])
            print(similarity.shape)
        else:
            similarity = self.pair_cosin_score(query_feats, gallery_feats)

        top_inds = np.argsort(-similarity, axis=1)

        # calculate top1
        analysis_index_list = []
        analysis_gallery_index_list = []
        similarity_score_list = []
        confusion_type_name_list = []
        
        total_real_unknown_num = 0

        TAR_num = 0
        FRR_num = 0
        FAR_num = 0    
        TNR_num = 0        

        G_score_list = []
        I_score_list = []

        for index in range(query_num):
            j = top_inds[index, 0]
            sencond_j_list = top_inds[index, 1:]    

            for index_second_j in sencond_j_list:
                if enroll_id_list[j] != enroll_id_list[index_second_j]:
                    sencond_j = index_second_j
                    break
              
            if index in identity_id_list[j]:
                if MLS_flag:
                    if cos_similarity[index, j] > threshold:
                        TAR_num += 1
                        confusion_type_name_list.append('TAR')
                    # FRR
                    else:
                        FRR_num += 1
                        confusion_type_name_list.append('FRR')

                    G_score_list.append(cos_similarity[index, j])
                    similarity_score_list.append(cos_similarity[index, j])
                    similarity_score_list.append(cos_similarity[index, sencond_j])
                else:
                    if similarity[index, j] > threshold:
                        TAR_num += 1
                        confusion_type_name_list.append('TAR')
                    # FRR
                    else:
                        FRR_num += 1
                        confusion_type_name_list.append('FRR')
                    
                    G_score_list.append(similarity[index, j])
                    similarity_score_list.append(similarity[index, j])
                    similarity_score_list.append(similarity[index, sencond_j])
            else:                 
                in_database = False
                for identity_id_index_list in identity_id_list:
                    if index in identity_id_index_list:
                        in_database = True
                
                if MLS_flag:
                    if cos_similarity[index, j] > threshold:
                        FAR_num += 1 
                        confusion_type_name_list.append('FAR')                        
                else:
                    if similarity[index, j] > threshold:
                        FAR_num += 1    
                        confusion_type_name_list.append('FAR')

                if not in_database: 
                    total_real_unknown_num += 1  

                    if MLS_flag:
                        if cos_similarity[index, j] <= threshold:   
                            TNR_num += 1
                            confusion_type_name_list.append('TNR')          
                    else:       
                        if similarity[index, j] <= threshold:
                            TNR_num += 1
                            confusion_type_name_list.append('TNR')          
                else:
                    if MLS_flag:
                        if cos_similarity[index, j] <= threshold:   
                            FRR_num += 1
                            confusion_type_name_list.append('FRR')
                    else:       
                        if similarity[index, j] <= threshold:
                            FRR_num += 1
                            confusion_type_name_list.append('FRR')
            
                if MLS_flag:
                    I_score_list.append(cos_similarity[index, j])
                    I_score_list.append(cos_similarity[index, sencond_j])    
                    similarity_score_list.append(cos_similarity[index, j])
                    similarity_score_list.append(cos_similarity[index, sencond_j])                
                else:
                    I_score_list.append(similarity[index, j])
                    I_score_list.append(similarity[index, sencond_j])
                    similarity_score_list.append(similarity[index, j])
                    similarity_score_list.append(similarity[index, sencond_j])

            analysis_index_list.append(index)  
            analysis_gallery_index_list.append(j)
            analysis_gallery_index_list.append(sencond_j)

        assert len(confusion_type_name_list) == len(analysis_index_list)

        eps = 1e-6
        top1_tar = TAR_num / (query_num - total_real_unknown_num + eps)
        top1_far = FAR_num / (query_num + eps)
        top1_frr = FRR_num / (query_num - total_real_unknown_num + eps)
        top1_tnr = TNR_num / (total_real_unknown_num + eps)

        # claculate FDR
        if len(G_score_list) == 0:
            G_score_list.append(-1)
        if len(I_score_list) == 0:
            I_score_list.append(-1)

        positive_score_mean = np.mean(G_score_list)
        negative_score_mean = np.mean(I_score_list)
        positive_score_std = np.std(G_score_list)
        negative_score_std = np.std(I_score_list)

        FDR = ((positive_score_mean - negative_score_mean) ** 2) / (positive_score_std ** 2 + negative_score_std ** 2)

        return top1_tar, top1_far, top1_frr, top1_tnr, confusion_type_name_list, analysis_index_list, analysis_gallery_index_list, similarity_score_list, G_score_list, I_score_list, FDR, positive_score_mean, negative_score_mean

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]

        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size

        score_max = np.max(score_list)
        score_min = np.min(score_list)

        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min + step * np.array(range(1, num_thresholds + 1))

        fpr_list = []
        tpr_list = []

        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list > threshold) / pos_pair_nums
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)

        best_index = np.argmax(tpr - fpr)
        best_thres = threshold_list[best_index]

        return best_thres
