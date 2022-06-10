"""
@author: Jun Wang
@date: 20201012 
@contact: jun21wangustc@gmail.com
"""  

import argparse
import yaml
import os
from megaface.megaface_evaluator import CommonMegaFaceEvaluator
from megaface.megaface_evaluator import MaskedMegaFaceEvaluator

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='megaface test protocal in python.')
    conf.add_argument("--data_conf_file", type = str, 
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--max_rank", type = int, 
                      help = "Rank N accuray..")
    conf.add_argument("--facescrub_feature_dir", type = str, 
                      help = "The dir of facescrub features.")
    conf.add_argument("--megaface_feature_dir", type = str, 
                      help = "The dir of megaface features.")
    conf.add_argument("--masked_facescrub_feature_dir", type = str, 
                      help = "The dir of masked facescrub features.")    
    conf.add_argument("--is_concat", type = int, 
                      help = "If the feature is concated by two nomalized features.")
    conf.add_argument('--model_path', type=str, default='mv_epoch_8.pt',
                        help='The path of model')
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)['MegaFace']
        facescrub_json_list = data_conf['facescrub_list']
        megaface_json_list = data_conf['megaceface_list']
        megaface_mask = data_conf['megaface-mask']
        masked_croped_facescrub_folder = data_conf['masked_croped_facescrub_folder']
    is_concat = True if args.is_concat == 1 else False
    # get feature path by model name as path
    if os.path.isdir(args.model_path):
        model_name_list = os.listdir(args.model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                print('test %s' % model_name)
                model_path = os.path.join(args.model_path, model_name)
                method_name = os.path.basename(model_path)

                facescrub_feature_dir \
                    = os.path.join(args.facescrub_feature_dir, model_name)
                megaface_feature_dir \
                    = os.path.join(args.megaface_feature_dir, model_name)
                masked_facescrub_feature_dir \
                    = os.path.join(args.masked_facescrub_feature_dir, model_name)
                masked_facescrub_wrong_image_dir \
                    = os.path.join(args.masked_facescrub_feature_dir, model_name + "_wrong_images")

                if not os.path.exists(masked_facescrub_wrong_image_dir):
                    os.mkdir(masked_facescrub_wrong_image_dir)

                if megaface_mask == 0:
                    megaFaceEvaluator = CommonMegaFaceEvaluator(
                        facescrub_json_list, megaface_json_list,
                        facescrub_feature_dir, megaface_feature_dir,
                        is_concat)
                elif megaface_mask == 1:
                    megaFaceEvaluator = MaskedMegaFaceEvaluator(
                        facescrub_json_list, megaface_json_list,
                        facescrub_feature_dir, megaface_feature_dir,
                        masked_facescrub_feature_dir, masked_croped_facescrub_folder, masked_facescrub_wrong_image_dir, is_concat)
                megaFaceEvaluator.test_cmc(model_name, args.max_rank)
