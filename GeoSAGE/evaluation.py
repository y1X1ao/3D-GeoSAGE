import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def evaluate(model, data, epoch):
    """
    评估模型在验证集上的表现，并返回多项指标：准确率、AUC、精确率、召回率、F1分数
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        out = model(data)
        val_mask = data.val_mask
        valid_mask = data.y[val_mask] != -1  # 忽略无标签样本

        # 真实标签和预测概率
        y_true = data.y[val_mask][valid_mask].cpu().numpy()
        y_pred_probs = out[val_mask][valid_mask].softmax(dim=1)[:, 1].cpu().numpy()  # 类别1的概率
        y_pred = out[val_mask][valid_mask].max(dim=1)[1].cpu().numpy()  # 预测类别

        # 计算准确率
        correct = (y_pred == y_true).sum().item()
        total = valid_mask.sum().item()
        val_acc = correct / total if total > 0 else 0

        # 计算其他评估指标
        if total > 0:
            auc = roc_auc_score(y_true, y_pred_probs)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            auc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return val_acc, auc, precision, recall, f1

# 定义测试函数
def test(model, data):
    """测试模型在测试集上的表现，并打印调试信息"""
    model.eval()
    with torch.no_grad():
        # 模型前向传播
        out = model(data)
        mask = data.test_mask

        print(f"Test Mask Sum: {mask.sum().item()}") 
        print(f"Test Mask Labels: {data.y[mask]}")  
        print(f"Model Output Shape: {out.shape}")  

        # 筛选出有效的测试样本
        valid_mask = data.y[mask] != -1
        if valid_mask.sum().item() == 0:
            print("No valid test samples available!")
            return 0.0

        # 模型输出
        test_output = out[mask][valid_mask]
        test_labels = data.y[mask][valid_mask]

        # 打印预测和真实标签对比
        print(f"Test Output (first 5 samples): {test_output[:5]}")
        print(f"Test Labels (first 5 samples): {test_labels[:5]}")

        # 预测类别
        pred = test_output.max(1)[1]

        # 计算准确率
        correct = pred.eq(test_labels).sum().item()
        acc = correct / valid_mask.sum().item()

         # 使用 softmax 获取每个类的概率
        probabilities = torch.softmax(test_output, dim=1)[:, 1]  # 只取正类的概率

        pred_np = pred.cpu().numpy()  # 将Tensor转换为numpy数组
        test_labels_np = test_labels.cpu().numpy()  # 将Tensor转换为numpy数组
        precision = precision_score(test_labels_np, pred_np, average='binary')
        recall = recall_score(test_labels_np, pred_np, average='binary')
        f1 = f1_score(test_labels_np, pred_np, average='binary')

        # 计算 AUC 分数
        probabilities_np = probabilities.cpu().numpy()  # 概率转换为 numpy
        auc_score = roc_auc_score(test_labels_np, probabilities_np)

         # 打印测试结果信息
        print(f"Correct Predictions: {correct}, Total Samples: {valid_mask.sum().item()}, Test Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}")
        
        # 计算并绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(test_labels_np, probabilities_np)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Test Set)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()  # 可以根据需要选择显示或保存
        return acc, precision, recall, f1, auc_score
    

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_rf_svm(model, X_test, y_test):
    """ 评估RF或SVM模型 """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}