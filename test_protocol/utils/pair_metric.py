import numpy as np
import os
import torch


def pair_cosin_score(x1, x2):
        cur_score = np.dot(x1, x2)
        return cur_score


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None, use_attention_only=False):
    x1, x2 = np.array(x1), np.array(x2)
    sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
    mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    cos_theta = np.sum(mu1*mu2, axis=-1)
    sigma_sq_mutual = sigma_sq_mutual.ravel()
    dist1 = 2*(1-cos_theta) / (1e-10 + sigma_sq_mutual)
    dist2 = np.log(sigma_sq_mutual)

    if use_attention_only:
        dist = dist1
    else:
        dist = dist1 + dist2
    return -dist


# reference by https://blog.csdn.net/SoftPoeter/article/details/86629329
def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def calculate_density(image_feature, anchor_embedding_set):
    k = 10
    image_feature = np.expand_dims(image_feature, axis=0)
    cos_similarity = np.dot(image_feature, anchor_embedding_set.transpose())
    assert np.max(cos_similarity, axis=1) <= 1.0
    cos_similarity_index = partition_arg_topK(-cos_similarity, k, axis=1)
    data_term = np.exp(cos_similarity[0][cos_similarity_index].reshape(-1))
    density = 1.0 / np.sum(data_term)

    return density

def pair_IDA_score(x1, x2, anchor_embedding_set):
    if x1.shape[-1] == 257 or x1.shape[-1] == 513:
        x1, x2 = np.array(x1), np.array(x2)
        D = int(x1.shape[-1] - 1)
        f1, density1 = x1[:D], x1[D:]
        f2, density2 = x2[:D], x2[D:]
    else:
        f1, f2 = np.array(x1), np.array(x2)
        density1 = calculate_density(f1, anchor_embedding_set)
        density2 = calculate_density(f2, anchor_embedding_set)

    similarity = np.dot(x1, x2)
    DAO_score = np.exp(similarity) * (density1 + density2)
    return DAO_score