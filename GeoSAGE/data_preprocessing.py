import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    """加载数据并进行预处理"""
    data = pd.read_csv(file_path)
    coords = data[['X', 'Y', 'Z']].values
    features = data.iloc[:, 3:-1].values  # 选择地球化学特征（剔除标签和坐标）
    labels = data['label'].fillna(-1).values
    return coords, features, labels

def standardize_features(features):
    """标准化地球化学特征"""
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def to_torch_tensors(features, labels, coords):
    """将数据转换为PyTorch张量格式"""
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    coords = torch.tensor(coords, dtype=torch.float)
    return features, labels, coords


# 有标签数据划分为训练集、验证集和测试集
# 假设 coords 是三维地学数据的空间坐标 (X, Y, Z)
# 可以根据 coords 进行区域划分，将整个研究区分成若干个子区域
def spatial_partitioning(coords, num_partitions=5):
    # 将研究区划分为 num_partitions 个子区域
    # 根据子区域的索引进行分区，例如根据 X 坐标等值线进行分区
    x_partitions = np.linspace(coords[:, 0].min(), coords[:, 0].max(), num_partitions + 1)
    region_indices = np.digitize(coords[:, 0], bins=x_partitions) - 1
    return region_indices

# 根据区域索引进行严格的训练集、验证集和测试集划分
def strict_spatial_partitioning(region_indices, num_regions):
    """
    根据区域索引进行严格的训练、验证和测试集划分，确保不同数据集之间在空间上互不重叠
    :param region_indices: 每个样本的区域索引
    :param num_regions: 总的区域数量
    :return: 训练集、验证集和测试集的索引
    """
    # 将区域按照一定的比例分配给训练集、验证集和测试集
    region_ids = np.arange(num_regions)
    
    # 假设将前 60% 的区域分配给训练集，中间 20% 分配给验证集，最后 20% 分配给测试集
    train_regions = region_ids[:int(0.6 * num_regions)]
    val_regions = region_ids[int(0.6 * num_regions):int(0.8 * num_regions)]
    test_regions = region_ids[int(0.8 * num_regions):]

    # 根据区域分配采样点的索引
    train_indices = np.where(np.isin(region_indices, train_regions))[0]
    val_indices = np.where(np.isin(region_indices, val_regions))[0]
    test_indices = np.where(np.isin(region_indices, test_regions))[0]

    return train_indices, val_indices, test_indices