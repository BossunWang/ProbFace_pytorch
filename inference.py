from model import backbone, uncertainty_head
import torch
import os
import numpy as np


adaface_models = {
    'ir_50': "pretrained/adaface_ir50_webface4m.ckpt",
    'ir_101': "pretrained/adaface_ir101_webface12m.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = backbone.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor


if __name__ == '__main__':
    import time

    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model('ir_101')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        start = time.time()
        feature, feature_fusion, norm = model(torch.randn(batch_size, 3, 112, 112).to(device))
        end = time.time()
        print("fw time: ", end - start)
        print("feature:", feature.size())
        print("feature_fusion:", feature_fusion.size())

    unh = uncertainty_head.UncertaintyHead(in_feat=feature_fusion.size(-1)).to(device)
    unh.train()

    log_sigma_sq = unh(feature_fusion)
    print("log_sigma_sq:", log_sigma_sq)
