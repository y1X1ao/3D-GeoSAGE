import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from predict import predict_and_export
# from graph_building import build_knn_graph
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv(r'D:\OneDrive\paper_code\GNN\code\早子沟3D数据.csv')

# 选择XYZ坐标与地球化学特征
coords = data[['X', 'Y', 'Z']].values
features = data.iloc[:, 3:-1].values  # 选择地球化学特征

# 筛选有标签的数据（0 或 1）
labeled_data = data.dropna(subset=['label'])
coords_labeled = labeled_data[['X', 'Y', 'Z']].values
features_labeled = labeled_data.iloc[:, 3:-1].values  # 地球化学特征
labels_labeled = labeled_data['label'].values  # 筛选非空标签

# 根据坐标进行空间分区
def spatial_partitioning(coords, num_partitions=5):
    x_partitions = np.linspace(coords[:, 0].min(), coords[:, 0].max(), num_partitions + 1)
    region_indices = np.digitize(coords[:, 0], bins=x_partitions) - 1
    return region_indices

# 根据区域索引进行严格的训练集、验证集和测试集划分
def strict_spatial_partitioning(region_indices, num_regions):
    region_ids = np.arange(num_regions)
    train_regions = region_ids[:int(0.6 * num_regions)]
    val_regions = region_ids[int(0.6 * num_regions):int(0.8 * num_regions)]
    test_regions = region_ids[int(0.8 * num_regions):]

    train_indices = np.where(np.isin(region_indices, train_regions))[0]
    val_indices = np.where(np.isin(region_indices, val_regions))[0]
    test_indices = np.where(np.isin(region_indices, test_regions))[0]
    return train_indices, val_indices, test_indices

# 使用空间分区策略划分数据集
num_partitions = 5
region_indices = spatial_partitioning(coords_labeled, num_partitions)
train_indices, val_indices, test_indices = strict_spatial_partitioning(region_indices, num_partitions)

# 划分数据集
X_train = features_labeled[train_indices]
y_train = labels_labeled[train_indices]
X_val = features_labeled[val_indices]
y_val = labels_labeled[val_indices]
X_test = features_labeled[test_indices]
y_test = labels_labeled[test_indices]

# 对训练集进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 使用相同的Scaler对验证集和测试集进行标准化
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 将数据转为PyTorch格式
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 训练 RF 模型
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# 评估 RF 模型
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"RF Accuracy: {rf_acc}, Precision: {rf_prec}, Recall: {rf_recall}, F1 Score: {rf_f1}")

# 特征重要性分析
feature_importances = rf_model.feature_importances_
feature_names = data.columns[3:-1].values  # 对应的特征名

# 绘制特征重要性
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importances)[::-1]
plt.bar(range(X_train.shape[1]), feature_importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for Random Forest Model')
plt.tight_layout()
plt.show()

# 使用RF模型对原始数据进行预测并输出结果
predict_and_export(rf_model, coords, features, 'rf_predictions.csv', scaler)
