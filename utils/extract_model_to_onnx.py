import torch
import torch.nn.functional as F
import sys
import onnxruntime
import numpy as np
import os
import onnx

sys.path.append('../')
from model import backbone
from model.uncertainty_head import UncertaintyHead


class ModelLoader:
    """Load a model by network and weights file.

    Attributes:
        model(object): the model definition file.
        device(object): the machine's device whether to CPU or GPU.
    """

    def __init__(self, backbone_path, backbone_type, device):
        self.backbone_path = backbone_path
        self.model = self.load_backbone_pretrained_model(backbone_type)
        in_feat = 47040
        self.uncertainty_model = UncertaintyHead(in_feat=in_feat)
        self.device = device

    def load_backbone_pretrained_model(self, architecture='ir_101'):
        # load model and pretrained statedict
        model = backbone.build_model(architecture)
        statedict = torch.load(self.backbone_path, map_location=lambda storage, loc: storage)['state_dict']
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

        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)  # load checkpoint
        state_dict = ckpt['state_dict']
        self.uncertainty_model.load_state_dict(state_dict, strict=True)  # load
        uncertainty_model = self.uncertainty_model.to(self.device)
        return model, uncertainty_model


class ModelWrapper(torch.nn.Module):
    def __init__(self, backbone, head):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, image):
        feat, feature_fusions = self.backbone(image)
        log_sigma_sq = self.head(feature_fusions)
        sigma_x = torch.exp(log_sigma_sq)
        feat = F.normalize(feat)
        features = torch.cat([feat, sigma_x], dim=1)
        return features


def check_onnx_model(backbone_path, head_path, model_onnx_path):
    print("check_onnx_model:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    model_loader = ModelLoader(backbone_path, 'ir_101_v2', device)
    backbone, head = model_loader.load_model(head_path, backbone_only=False)
    model = ModelWrapper(backbone, head)
    model.eval()

    dummy_output = model(dummy_input)
    output_torch_to_np = dummy_output.cpu().data.numpy().reshape(-1)

    ort_session = onnxruntime.InferenceSession(model_onnx_path)
    dummy_input_np = dummy_input.cpu().data.numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}

    keys = list(ort_inputs.keys())
    ort_inputs[keys[0]] = ort_inputs[keys[0]].reshape(1, 3, 112, 112)
    ort_outs = ort_session.run(['output'], ort_inputs)
    output_onnx_to_np = np.array(ort_outs).reshape(-1)

    print("result:", (output_torch_to_np - output_onnx_to_np).mean())


def transform_to_onnx(backbone_path, head_path, model_onnx_path):
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    # check modified forward function for onnx exclude condition adapooling
    model_loader1 = ModelLoader(backbone_path, 'ir_101', device)
    backbone1, head1 = model_loader1.load_model(head_path, backbone_only=False)
    model1 = ModelWrapper(backbone1, head1)
    model1.eval()
    features_v1 = model1(dummy_input)

    model_loader2 = ModelLoader(backbone_path, 'ir_101_v2', device)
    backbone2, head2 = model_loader2.load_model(head_path, backbone_only=False)
    model2 = ModelWrapper(backbone2, head2)
    model2.eval()
    features_v2 = model2(dummy_input)

    difference = (features_v1 - features_v2).mean()
    print("check difference between model forward:", difference)

    torch.onnx.export(model2
                      , dummy_input
                      , model_onnx_path
                      , verbose=False
                      , opset_version=12
                      , input_names=['input']
                      , output_names=['output'])
    # Checks
    onnx_model = onnx.load(model_onnx_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model


def main():
    backbone_path = '../weights/adaface_ir101_webface12m.ckpt'
    head_path = '../prob_face_ir_101_masked_final/prob_face_uc.pt'
    model_onnx_path = "../onnx_weights/prob_face_uc.onnx"
    transform_to_onnx(backbone_path, head_path, model_onnx_path)

    print("Export of torch model to onnx complete!")

    check_onnx_model(backbone_path, head_path, model_onnx_path)


if __name__ == "__main__":
    main()
