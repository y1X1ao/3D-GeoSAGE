import torch
import pandas as pd
import numpy as np


def predict_unlabeled_samples(model, data, coords):
    """对无标签的样本进行预测，并输出坐标和预测为正样本的概率"""
    model.eval()
    with torch.no_grad():
        out = model(data)
        prob = torch.softmax(out, dim=1)[:, 1]
        coords_np = coords.cpu().numpy()
        result = pd.DataFrame(coords_np, columns=["X", "Y", "Z"])
        result["Positive_Probability"] = prob.cpu().numpy()

        return result
    

def predict_and_export(model, coords, features, filename,scaler):
   
    with torch.no_grad():  
        features_scaled = scaler.transform(features)  
        proba = model.predict_proba(features_scaled)[:, 1]  
        output = np.column_stack((coords, proba))  
        df_output = pd.DataFrame(output, columns=['X', 'Y', 'Z', 'Predicted Probability'])
        df_output.to_csv(filename, index=False)