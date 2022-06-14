""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import argparse
import yaml
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torchvision import transforms
from lfw.pairs_parser import PairsParserFactory
from test_1N.evaluator_1N import Evaluator1N
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor

sys.path.append('..')
from data.test_dataset import TestDataset1N, TestDataset1NFromJason


def accu_key(elem):
    return elem[2]


def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'a') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='lfw test protocal.')
    conf.add_argument("--test_set", type=str,
                      help="lfw, cplfw, calfw, agedb, rfw_African, \
                      rfw_Asian, rfw_Caucasian, rfw_Indian.")
    conf.add_argument("--load_feature", type=bool, default=False)
    conf.add_argument("--data_conf_file", type=str,
                      help="the path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type=str,
                      help="Resnet, Mobilefacenets..")
    conf.add_argument("--backbone_conf_file", type=str,
                      help="The path of backbone_conf.yaml.")
    conf.add_argument("--load_backbone_only", action='store_true', default=False)
    conf.add_argument('--batch_size', type=int, default=1024)
    conf.add_argument('--threshold', type=float, default=0.5)
    conf.add_argument('--from_json', type=bool, default=False)
    conf.add_argument('--model_path', type=str, default='mv_epoch_8.pt',
                      help='The path of model or the directory which some models in.')
    conf.add_argument('--mean', type=float, default=127.5)
    conf.add_argument('--std', type=float, default=128)
    args = conf.parse_args()
    # parse config.
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)[args.test_set]
        pairs_file_path = data_conf['pairs_file_path']
        croped_face_folder = data_conf['croped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']

    print('test name:', args.test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.from_json:
        enroll_data_loader = DataLoader(TestDataset1NFromJason(os.path.join(croped_face_folder, 'enroll'), False)
                                        , batch_size=args.batch_size
                                        , num_workers=4, shuffle=False)
        identity_data_loader = DataLoader(TestDataset1NFromJason(os.path.join(croped_face_folder, 'identity'), False)
                                          , batch_size=args.batch_size
                                          , num_workers=4, shuffle=False)
        feature_extractor = CommonExtractor(device)
        evaluator_1N = Evaluator1N(enroll_data_loader, identity_data_loader, feature_extractor)
        top1_tar, top1_far, top1_frr, top1_tnr, FDR, positive_score_mean, negative_score_mean \
            = evaluator_1N.test(None, args.threshold,
                                '1N_features/' + 'json_' + str(args.threshold) + '_' + args.test_set
                                , args.load_feature
                                , from_json=args.from_json)
        accu_list = [(
            args.test_set, os.path.basename(args.model_path), args.threshold, top1_tar, top1_far, top1_frr,
            top1_tnr, FDR, positive_score_mean, negative_score_mean)]
    else:
        # define dataloader
        enroll_data_loader = DataLoader(TestDataset1N(os.path.join(croped_face_folder, 'enroll'), False, args.mean, args.std)
                                        , batch_size=args.batch_size
                                        , num_workers=4, shuffle=False)
        identity_data_loader = DataLoader(TestDataset1N(os.path.join(croped_face_folder, 'identity'), False, args.mean, args.std)
                                          , batch_size=args.batch_size
                                          , num_workers=4, shuffle=False)
        # model def
        model_loader = ModelLoader(args.backbone_type, device)
        feature_extractor = CommonExtractor(device)
        evaluator_1N = Evaluator1N(enroll_data_loader, identity_data_loader, feature_extractor)
        if os.path.isdir(args.model_path):
            accu_list = []
            model_name_list = os.listdir(args.model_path)
            for model_name in model_name_list:
                if model_name.endswith('.pt'):
                    print('test %s' % model_name)
                    model_path = os.path.join(args.model_path, model_name)
                    model = model_loader.load_model(model_path)
                    top1_tar, top1_far, top1_frr, top1_tnr, FDR, positive_score_mean, negative_score_mean \
                        = evaluator_1N.test(model, args.threshold, '1N_features/' + model_name + '_' + str(
                        args.threshold) + '_' + args.test_set
                                            , args.load_feature)
                    accu_list.append((args.test_set, os.path.basename(model_path), args.threshold, top1_tar, top1_far,
                                      top1_frr, top1_tnr, FDR, positive_score_mean, negative_score_mean))
            accu_list.sort(key=accu_key, reverse=True)
        else:
            print('test only one model')
            model = model_loader.load_model(args.model_path)
            top1_tar, top1_far, top1_frr, top1_tnr, FDR, positive_score_mean, negative_score_mean \
                = evaluator_1N.test(model, args.threshold, '1N_features/' + args.model_path.split('/')[-1] + '_' + str(
                args.threshold) + '_' + args.test_set
                                    , args.load_feature)
            accu_list = [(
                         args.test_set, os.path.basename(args.model_path), args.threshold, top1_tar, top1_far, top1_frr,
                         top1_tnr, FDR, positive_score_mean, negative_score_mean)]

    pretty_tabel = PrettyTable(
        ["test set", "model_name", "threshold", "top1 TAR", "top1 FAR", "top1 FRR", "top1 TNR", "FDR", "G_mean",
         "I_mean"])
    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)

    ptable_to_csv(pretty_tabel, os.path.join('log', args.test_set + '.csv'))
