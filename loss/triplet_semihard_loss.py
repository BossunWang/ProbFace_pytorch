import torch
import torch.nn as nn


def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()


class TripletSemiHardLoss(nn.Module):
    """
    the same with tf.triplet_semihard_loss
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self):
        super(TripletSemiHardLoss, self).__init__()

    def masked_maximum(self, data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the maximum.
            Returns:
              masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
            """
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
        return masked_maximums

    def masked_minimum(self, data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the minimum.
            Returns:
              masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
            """
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
        return masked_minimums

    def forward(self, pdist_matrix, target, margin=1.0, squared=True):
        """
        :param features: [B * N features]
        :param target: [B]
        :param square: if the distance squared or not.
        :return:
        """
        lshape = target.shape
        assert len(lshape) == 1
        labels = target.int().unsqueeze(-1)  # [B, 1]
        adjacency = labels == torch.transpose(labels, 0, 1)

        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        # Compute the mask

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
            torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                                   0.0, [batch_size, batch_size])
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # negatives_inside: largest D_an.
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.float() - torch.diag(cudafy(torch.ones([batch_size])))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = torch.sum(mask_positives).clamp(min=1.0)

        triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)

        # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
        return triplet_loss


if __name__ == '__main__':
    from model.uncertainty_head import UncertaintyHead
    from MLSloss import MLSLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 3
    feat_dim = 512
    in_feat = 47040
    unh = UncertaintyHead(in_feat).to(device)
    MLS = MLSLoss()
    TripletSemiHard = TripletSemiHardLoss()
    unh.train()

    uncertainty_module_output_size = 1

    feat = torch.randn(batch_size, feat_dim).to(device)
    norm = torch.norm(feat, 2, 1, True)
    feat = torch.div(feat, norm)
    feat_fusion = torch.randn(batch_size, in_feat).to(device)
    log_sigma_sq = unh(feat_fusion)

    label = torch.randint(0, 3, (batch_size,)).to(device)
    # label = torch.Tensor([2, 1, 0]).to(device)
    print("label:", label)
    _, attention_mat, mean_pos, mean_neg = MLS(feat.detach(), log_sigma_sq, label)
    print("attention_mat:", attention_mat)

    triplet_loss = TripletSemiHard(attention_mat, label, margin=3.0)
    print("triplet_loss", triplet_loss)
    triplet_loss.backward()
