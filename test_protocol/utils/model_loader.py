"""
@author: Jun Wang
@date: 20201016
@contact: jun21wangustc@gmail.com
"""

import torch
import sys

sys.path.append("../../")
from model import backbone
from model.backbone import IR_101
from model.uncertainty_head import UncertaintyHead


class ModelLoader:
    """Load a model by network and weights file.

    Attributes: 
        model(object): the model definition file.
        device(object): the machine's device whether to CPU or GPU.
    """

    def __init__(self, backbone_type, device):
        self.adaface_models = {
            'ir_50': "../pretrained/adaface_ir50_webface4m.ckpt",
            'ir_101': "../pretrained/adaface_ir101_webface12m.ckpt",
        }
        self.model = self.load_backbone_pretrained_model(backbone_type)
        in_feat = 47040
        self.uncertainty_model = UncertaintyHead(in_feat=in_feat)
        self.device = device

    def load_backbone_pretrained_model(self, architecture='ir_50'):
        # load model and pretrained statedict
        assert architecture in self.adaface_models.keys()
        model = backbone.build_model(architecture)
        statedict = torch.load(self.adaface_models[architecture])['state_dict']
        model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
        model.load_state_dict(model_statedict)
        return model

    def load_model(self, model_path, backbone_only=False):
        """The custom method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        model = self.model.to(self.device)

        if backbone_only:
            return model

        print('load uncertainty_head')
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)  # load checkpoint
        state_dict = ckpt['state_dict']
        self.uncertainty_model.load_state_dict(state_dict, strict=True)  # load
        uncertainty_model = self.uncertainty_model.to(self.device)
        model_list = [model, uncertainty_model, 'ProbFace']
        return model_list
