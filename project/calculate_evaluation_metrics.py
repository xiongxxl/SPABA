import torch
import math
from atom_criterion import get_atom_error

def compute_metrics_tensor(smiles,y_true, y_pred):
    # 转为 long 型 tensor，确保是 0/1
    y_true = y_true.long()
    y_pred = y_pred.long()

    TP = torch.sum((y_true == 1) & (y_pred == 1)).item()
    TN = torch.sum((y_true == 0) & (y_pred == 0)).item()
    FP = torch.sum((y_true == 0) & (y_pred == 1)).item()
    FN = torch.sum((y_true == 1) & (y_pred == 0)).item()
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc_denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN - FP * FN) / mcc_denominator) if mcc_denominator > 0 else 0

    val_dict={
                    # 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                      'smiles':[smiles],
                    'Accuracy': [accuracy],
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1-score': [f1_score],
                    'MCC': [mcc]
              }



    return val_dict


if __name__=="__main__":
# 示例用法
    y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0])
    smiles='CCCCCC'
    metrics = compute_metrics_tensor(smiles,y_true, y_pred)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")