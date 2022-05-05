import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSLoss(nn.Module):

    def __init__(self):
        super(MLSLoss, self).__init__()

    def negMLS(self, mu_X, sigma_sq_X):
        cos_theta = torch.matmul(mu_X, mu_X.T)
        # sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
        sig_sum = sigma_sq_X + sigma_sq_X.T
        # diffs = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
        diff = 2 * (1 - cos_theta) / (1e-10 + sig_sum) + torch.log(sig_sum)
        attention_mat = 2 * (1 - cos_theta) / (1e-10 + sig_sum)
        return diff, attention_mat

    def forward(self, mu_X, log_sigma_sq, gty):
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int().to(gty.device)
        sig_X = torch.exp(log_sigma_sq)

        loss_mat, attention_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        # print("non_diag_mask:", non_diag_mask)
        # print("gty_mask:", gty_mask)
        # print("not gty_mask:", 1 - gty_mask)
        pos_mask = (non_diag_mask * gty_mask) > 0
        neg_mask = (non_diag_mask * (1 - gty_mask)) > 0
        # print("pos_mask:", pos_mask)
        # print("neg_mask:", neg_mask)
        pos_loss = loss_mat[pos_mask].mean()
        mean_pos = attention_mat[pos_mask].mean()
        mean_neg = attention_mat[neg_mask].mean()
        return pos_loss, attention_mat, mean_pos, mean_neg


if __name__ == '__main__':
    from model.uncertainty_head import UncertaintyHead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 3
    feat_dim = 512
    in_feat = 47040
    unh = UncertaintyHead(in_feat).to(device)
    MLS = MLSLoss()
    unh.train()

    feat = torch.randn(batch_size, feat_dim).to(device)
    norm = torch.norm(feat, 2, 1, True)
    feat = torch.div(feat, norm)
    feat_fusion = torch.randn(batch_size, in_feat).to(device)

    log_sigma_sq = unh(feat_fusion)
    print("log_sigma_sq:", log_sigma_sq)

    label = torch.randint(0, 2, (batch_size, )).to(device)
    print("label:", label)
    MLS_loss, attention_mat, mean_pos, mean_neg = MLS(feat.detach(), log_sigma_sq, label)
    print("MLS loss:", MLS_loss)
    print("attention_mat:", attention_mat)
    MLS_loss.backward()