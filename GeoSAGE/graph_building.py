import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix

def build_knn_graph(coords, k=10):
    """基于坐标构建KNN图"""
    adj_matrix = kneighbors_graph(coords, k, mode='distance', include_self=False)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
    return edge_index, edge_attr

def positional_encoding(coords, max_freq=10, num_freqs=6):
    """
    使用三角函数对位置进行编码，增强空间信息
    :param coords: 样本的空间坐标
    :param max_freq: 最大频率
    :param num_freqs: 使用的频率数
    :return: 位置编码
    """
    freqs = np.logspace(0, np.log10(max_freq), num_freqs)
    encodings = []
    for freq in freqs:
        for axis in range(coords.shape[1]):
            encodings.append(np.sin(freq * coords[:, axis]))
            encodings.append(np.cos(freq * coords[:, axis]))
    return np.vstack(encodings).T  # 转置确保维度为 (样本数, 特征数)

def augment_features_with_position(features, positional_enc):
    """合并地球化学特征和位置编码"""
    return torch.cat([features, positional_enc], dim=1)