import torch
from utils import EarlyStopping
from evaluation import evaluate
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


def generate_pseudo_labels(model, data, threshold=0.95):
    """
    生成伪标签，用于弱监督学习。
    :param model: 当前训练好的模型
    :param data: 图数据对象（包含所有节点和边特征）
    :param threshold: 伪标签置信度阈值
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        out = model(data)  
        preds = torch.softmax(out, dim=1) 
        pseudo_labels = preds.max(dim=1)[1]  
        confidences = preds.max(dim=1)[0]  

    new_labels = data.y.clone()
    no_label_mask = data.y == -1 & data.train_mask 
    high_conf_mask = confidences > threshold  # 筛选置信度高于阈值的样本

    new_labels[no_label_mask & high_conf_mask] = pseudo_labels[no_label_mask & high_conf_mask]
    print(f"伪标签生成：新增伪标签样本数：{(no_label_mask & high_conf_mask).sum().item()}")
    return new_labels

def calculate_metrics(preds, labels):
    """计算 AUC、Recall、Precision 和 F1 分数"""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if len(set(labels_np)) == 1:
        auc = 0.0
    else:
        auc = roc_auc_score(labels_np, preds_np[:, 1])
    
    preds_class = preds_np.argmax(axis=1)
    recall = recall_score(labels_np, preds_class, zero_division=0)
    precision = precision_score(labels_np, preds_class, zero_division=0)
    f1 = f1_score(labels_np, preds_class, zero_division=0)
    
    return auc, recall, precision, f1





def train_with_early_stopping(model, data, optimizer, loss_fn, writer, patience=10, n_epochs=100, pseudo_label_freq=20,generate_pseudo=False):
    """
    使用早停机制进行训练，并加入伪标签模块。
    :param model: 图神经网络模型
    :param data: 图数据对象（包含所有节点和边特征）
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :param writer: TensorBoard 日志记录器
    :param patience: 早停机制耐心度
    :param n_epochs: 最大训练轮数
    :param pseudo_label_freq: 伪标签更新频率（每隔多少个 epoch 更新一次伪标签）
    :return: 训练好的模型
    """
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        # 每隔一定轮数更新伪标签
        if generate_pseudo == True and epoch % pseudo_label_freq == 0 and epoch > 0:
            # 生成新的伪标签并更新到 data.y 中
            data.y = generate_pseudo_labels(model, data, threshold=0.8)

        # 模型训练
        out = model(data)
        mask = data.train_mask &(data.y != -1) 
        loss = loss_fn(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()

        # 记录训练损失
        writer.add_scalar('Loss', loss.item(), epoch)

        # 验证集评估
        # val_acc = evaluate(model, data, epoch)
        # 验证集评估
        val_acc, auc, precision, recall, f1 = evaluate(model, data, epoch)

        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {val_acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

        

        # print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

        # 早停检查
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载最佳模型权重
    model.load_state_dict(torch.load('best_model.pt'))
    writer.close()
    return model
