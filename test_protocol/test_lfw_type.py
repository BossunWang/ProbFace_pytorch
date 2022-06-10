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
import matplotlib.pyplot as plt
import seaborn as sns

from lfw.pairs_parser import PairsParserFactory
from lfw.lfw_evaluator import LFWEvaluator
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('../')
from data.test_dataset import CommonTestDataset


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
    conf.add_argument("--offline_mode", type=bool, default=False)
    conf.add_argument("--load_feature", type=bool, default=True)
    conf.add_argument("--data_conf_file", type=str,
                      help="the path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type=str,
                      help="Resnet, Mobilefacenets..")
    conf.add_argument("--load_backbone_only", action='store_true', default=False)
    conf.add_argument('--batch_size', type=int, default=1024)
    conf.add_argument('--model_path', type=str, default='mv_epoch_8.pt',
                      help='The path of model or the directory which some models in.')
    conf.add_argument('--anchor_embedding_path', type=str, default=None,
                      help='The path of anchor embedding set for Inter-class Discrepancy')
    conf.add_argument('--mean', type=float, default=127.5)
    conf.add_argument('--std', type=float, default=127.5)
    args = conf.parse_args()
    args.offline_mode = False
    # parse config.
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)[args.test_set]
        pairs_file_path = data_conf['pairs_file_path']
        croped_folder_list = data_conf['croped_face_folder']
        croped_folder_list = croped_folder_list.split(',')
        if len(croped_folder_list) == 2:
            croped_face_folder_p1 = croped_folder_list[0]
            croped_face_folder_p2 = croped_folder_list[1]
        else:
            croped_face_folder_p1 = croped_folder_list[0]
            croped_face_folder_p2 = croped_folder_list[0]

        image_list_file_path = data_conf['image_list_file_path']

    print('test name:', args.test_set)
    # define pairs_parser_factory
    pairs_parser_factory = PairsParserFactory(croped_face_folder_p1, croped_face_folder_p2, pairs_file_path, args.test_set)
    pairs_parser = pairs_parser_factory.get_parser()
    pair_list = pairs_parser.parse_pairs()
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(croped_face_folder_p1, croped_face_folder_p2, pair_list, False, args.mean, args.std),
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    # model def
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(args.backbone_type, device)
    feature_extractor = CommonExtractor(device)
    lfw_evaluator = LFWEvaluator(data_loader, pair_list, feature_extractor, args.anchor_embedding_path)
    
    IDA_file_name = ''
    if args.anchor_embedding_path is not None:
        IDA_file_name = '_' + args.anchor_embedding_path.split('/')[-1].split('.')[0]

    if os.path.isdir(args.model_path):
        accu_list = []
        model_name_list = os.listdir(args.model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                print('test %s' % model_name)
                model_path = os.path.join(args.model_path, model_name)
                model = model_loader.load_model(model_path, args.load_backbone_only)

                method_name = os.path.basename(model_path)
                save_dir = '11_features/' + model_name + '_' + args.test_set
                method_name += IDA_file_name
                save_dir += IDA_file_name

                mean, std, FDR, G_mean, I_mean, mean_thres, all_positive_score_list, all_negative_score_list \
                    = lfw_evaluator.test(model, save_dir, args.offline_mode, args.load_feature)
                accu_list.append((args.test_set, method_name, mean, std, FDR, G_mean, I_mean, mean_thres))

                # plot
                plot_img_filename = os.path.join('log', args.test_set + '_' + model_name + '.png')
                plt.figure(args.test_set + '_' + model_name)
                sns.distplot(all_positive_score_list, label="G")
                sns.distplot(all_negative_score_list, label="I")
                plt.axvline(mean_thres, 0, 1)
                plt.legend()
                plt.savefig(plot_img_filename)
                plt.close("all")

        accu_list.sort(key=accu_key, reverse=True)
    else:
        print('test only one model')
        model = model_loader.load_model(args.model_path)

        method_name = os.path.basename(args.model_path)
        save_dir = '11_features/' + args.model_path.split('/')[-1] + '_' + args.test_set
        method_name += IDA_file_name
        save_dir += IDA_file_name

        mean, std, FDR, G_mean, I_mean \
            = lfw_evaluator.test(model, save_dir
                                 , args.offline_mode, args.load_feature)
        accu_list = [(args.test_set, method_name, mean, std, FDR, G_mean, I_mean)]

    pretty_tabel = PrettyTable(["test set", "model_name", "mean accuracy", "standard error", "FDR", "G_mean", "I_mean", "mean_thres"])
    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)

    csv_file_name = args.test_set

    ptable_to_csv(pretty_tabel, os.path.join('log', csv_file_name + '.csv'))