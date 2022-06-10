"""
@author: Haoran Jiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import numpy as np
sys.path.append('../')
from utils.pair_metric import pair_cosin_score, pair_MLS_score, pair_IDA_score


class LFWEvaluator(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        pair_list(list): the pair list given by PairsParser.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, data_loader, pair_list, feature_extractor, anchor_embedding_path):
        """Init LFWEvaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            pair_list(list): the pair list given by PairsParser.
            feature_extractor(object): a feature extractor.
            anchor_embedding_path: the path of anchor embedding set
        """
        self.data_loader = data_loader
        self.pair_list = pair_list
        self.feature_extractor = feature_extractor
        self.cv_times = 10
        # load anchor embedding set
        self.anchor_embedding_set = None
        if anchor_embedding_path is not None:
            self.anchor_embedding_set = np.load(anchor_embedding_path)
            print('anchor_embedding_set:', self.anchor_embedding_set.shape)
        print('pair_list:', (len(self.pair_list) // self.cv_times) * self.cv_times)

    def test(self, model, save_dir=None, offline=False, only_load=False):
        if only_load:
            image_name2feature = self.feature_extractor.load_feature(save_dir + '_features/')
        elif offline:
            self.feature_extractor.extract_offline(save_dir + '_features/', model, self.data_loader)
            image_name2feature = self.feature_extractor.load_feature(save_dir + '_features/')
        else:
            image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)

        mean, std, FDR, G_mean, I_mean, mean_thres, all_positive_score_list, all_negative_score_list\
            = self.test_one_model(self.pair_list, image_name2feature)
        return mean, std, FDR, G_mean, I_mean, mean_thres, all_positive_score_list, all_negative_score_list

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize=True):
        """Get the accuracy of a model.
        
        Args:
            test_pair_list(list): the pair list given by PairsParser. 
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
        # shape is (cross validation times, len(test_pair_list) // cross validation times)
        cv_part_size = len(test_pair_list) // self.cv_times
        drop_last = cv_part_size * self.cv_times
        subsets_score_list = np.zeros((self.cv_times, cv_part_size), dtype=np.float32)
        subsets_label_list = np.zeros((self.cv_times, cv_part_size), dtype=np.int8)
        for index, cur_pair in enumerate(test_pair_list[:drop_last]):
            cur_subset = index // cv_part_size
            cur_id = index % cv_part_size
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]
            subsets_label_list[cur_subset][cur_id] = label
            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]

            # MLS
            if feat1.shape[-1] == 513 and feat2.shape[-1] == 513:
                cur_score = pair_MLS_score(feat1[:512], feat2[:512], feat1[-1], feat2[-1])
            # SRT
            elif feat1.shape[-1] == 1024 and feat2.shape[-1] == 1024:
                cur_score = pair_cosin_score(feat1[:512], feat2[512:]) 
            else:
                if not is_normalize:
                    feat1 = feat1 / np.linalg.norm(feat1)
                    feat2 = feat2 / np.linalg.norm(feat2)
                # IDA
                if self.anchor_embedding_set is not None:
                    cur_score = pair_IDA_score(feat1, feat2, self.anchor_embedding_set)
                # cosine
                else:
                    cur_score = pair_cosin_score(feat1, feat2)

            subsets_score_list[cur_subset][cur_id] = cur_score

        subset_train = np.array([True] * self.cv_times)
        accu_list = []
        best_thres_list = []
        all_positive_score_list = []
        all_negative_score_list = []

        for subset_idx in range(self.cv_times):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()

            subset_train[subset_idx] = True
            best_thres = self.getThreshold(train_score_list, train_label_list)
            best_thres_list.append(best_thres)
            print('cv %d best threshold: %f', subset_idx, best_thres)

            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            all_positive_score_list.extend(positive_score_list)
            all_negative_score_list.extend(negtive_score_list)

            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list <= best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / cv_part_size)

        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(self.cv_times) # ddof=1, division 9.
        mean_thres = np.mean(best_thres_list)
        print('mean_thres: ', mean_thres)

        # claculate FDR
        positive_score_mean = np.mean(all_positive_score_list)
        negative_score_mean = np.mean(all_negative_score_list)
        positive_score_std = np.std(all_positive_score_list)
        negative_score_std = np.std(all_negative_score_list)

        FDR = ((positive_score_mean - negative_score_mean) ** 2) / (positive_score_std ** 2 + negative_score_std ** 2)

        return mean, std, FDR, positive_score_mean, negative_score_mean\
            , mean_thres, all_positive_score_list, all_negative_score_list

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
