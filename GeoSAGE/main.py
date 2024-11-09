import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from data_preprocessing import load_data, standardize_features, to_torch_tensors,spatial_partitioning,strict_spatial_partitioning
from graph_building import build_knn_graph, positional_encoding, augment_features_with_position
from models import GraphSAGENet
from training import train_with_early_stopping
from evaluation import test
from torch_geometric.data import Data
import numpy as np
from predict import predict_unlabeled_samples

# Preprocess data
coords, features, labels = load_data('早子沟3D数据.csv')
features = standardize_features(features)
features, labels, coords = to_torch_tensors(features, labels, coords)

# 构建图和位置编码
edge_index, edge_attr = build_knn_graph(coords)
pos_enc = positional_encoding(coords.numpy())
pos_enc = torch.tensor(pos_enc, dtype=torch.float)
features = augment_features_with_position(features, pos_enc)

# TensorBoard
writer = SummaryWriter(log_dir='./runs/graphsage_experiment')


# 有标签数据划分为训练集、验证集和测试集
# 假设 coords 是三维地学数据的空间坐标 (X, Y, Z),可以根据 coords 进行区域划分，比如将整个研究区分成若干个子区域
# 使用改进的严格区域分区策略进行训练集、验证集和测试集的划分
num_partitions = 5  # 将研究区分为5个子区域
region_indices = spatial_partitioning(coords, num_partitions)
train_indices, test_indices, val_indices = strict_spatial_partitioning(region_indices, num_partitions)

# 更新训练集、验证集和测试集的 mask
train_mask = torch.zeros(labels.size(), dtype=torch.bool)
val_mask = torch.zeros(labels.size(), dtype=torch.bool)
test_mask = torch.zeros(labels.size(), dtype=torch.bool)

train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

# 更新图数据对象中的 mask
graph_data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
graph_data.train_mask = train_mask
graph_data.val_mask = val_mask
graph_data.test_mask = test_mask



# Training
model = GraphSAGENet(in_channels=features.shape[1], hidden_channels=128, out_channels=2)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

model = train_with_early_stopping(model, graph_data, optimizer, loss_fn, writer,n_epochs=500,patience=50,pseudo_label_freq=30,generate_pseudo=True)

# Testing
acc, precision, recall, f1, auc = test(model, graph_data)

# # Predict
predicted_probabilities = predict_unlabeled_samples(model, graph_data, coords)
predicted_probabilities.to_csv("predicted_probabilities_1029_ZZG.csv", index=False)