import cv2
import numpy as np
import sys
import os
from skimage import transform as trans
import torch
sys.path.append('../')

sys.path.append('../../')
from model.backbone import IR_101
from model.uncertainty_head import UncertaintyHead


class Embedding:
    def __init__(self, model_path, uc_model_path, model_type, data_shape, batch_size=1, ctx_id=0):
        torch.backends.cudnn.benchmark = True
        image_size = (112, 112)
        self.image_size = image_size
        # ======= model =======#
        BACKBONE_DICT = {'IR_101': IR_101}

        BACKBONE = BACKBONE_DICT[model_type](image_size)
        in_feat = 47040
        unh = UncertaintyHead(in_feat=in_feat)
        print("=" * 60)
        print("{} Backbone Generated".format(model_type))
        print("=" * 60)
        if model_path:
            print("=" * 60)
            if os.path.isfile(model_path) and os.path.isfile(uc_model_path):
                print("Loading Backbone Checkpoint '{}'".format(model_path))
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
                if "backbone" in checkpoint:                
                    BACKBONE.load_state_dict(checkpoint['backbone'])
                elif "state_dict" in checkpoint:                
                    # BACKBONE.load_state_dict(checkpoint['state_dict'])
                    statedict = checkpoint['state_dict']
                    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
                    BACKBONE.load_state_dict(model_statedict)
                else:
                    BACKBONE.load_state_dict(checkpoint)
                print("Loading Head Checkpoint '{}'".format(uc_model_path))
                ckpt = torch.load(uc_model_path, map_location=lambda storage, loc: storage)  # load checkpoint
                state_dict = ckpt['state_dict']
                unh.load_state_dict(state_dict, strict=True)  # load
            else:
                print("No Checkpoint Found at '{}'".format(model_path))
                exit()
            print("=" * 60)

        # if len(gpu_ids) > 1:
            # # multi-GPU setting
            # BACKBONE = nn.DataParallel(BACKBONE)

        BACKBONE.cuda()
        BACKBONE.eval()  # switch to evaluation mode
        unh.cuda()
        unh.eval()

        #self.image_size = input_size
        self.model = BACKBONE
        self.uc_head = unh
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob
    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs=torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, feature_fusions = self.model(imgs)
        log_sigma_sq = self.uc_head(feature_fusions)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        log_sigma_sq = log_sigma_sq.reshape([self.batch_size, 2 * log_sigma_sq.shape[1]])
        return feat.cpu().numpy(), log_sigma_sq.cpu().numpy()

