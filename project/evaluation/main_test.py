from torch.utils.data import Dataset
from calculate_evaluation_metrics import compute_metrics_tensor
import pandas as pd
from atom_criterion import get_atom_error
import os
from data_loader import SmilesNpyDataset
from torch.utils.data import DataLoader
import torch
from model import SmilesModel


def evaluate_model(model, loader,device,max_atom):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    df_metrics_dict_single = pd.DataFrame()
    df_predicted_atoms=pd.DataFrame()
    with torch.no_grad():
        for inputs, labels, smi_length,smiles in loader:
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            threshold = 0.5
            outputs = torch.sigmoid(outputs)
            predicted = (outputs >= threshold).float()
            for i in range(labels.size(0)):
                label_atoms = labels[i][:smi_length[i]]
                predicted_atoms = predicted[i][:smi_length[i]]
                atom_error=get_atom_error(label_atoms,predicted_atoms)
                if 0<=atom_error<=max_atom:
                # if torch.equal(label_atoms, predicted_atoms):
                    correct += 1  # Count as correct
                total += labels.size(0)
                smiles=smiles[0]
                metrics_dict = compute_metrics_tensor(smiles,label_atoms, predicted_atoms)
                predicted_atoms_arry=predicted_atoms.numpy()
                predicted_atoms_dict={  'smiles':[smiles],
                                         'predicted_atoms': [predicted_atoms_arry],
                                          'label_atoms':[label_atoms]
                                      }
                predicted_atoms_dict=pd.DataFrame(predicted_atoms_dict)
                df_metrics_dict=pd.DataFrame(metrics_dict)
                df_predicted_atoms=pd.DataFrame(pd.concat([df_predicted_atoms,predicted_atoms_dict],ignore_index=True))
                df_metrics_dict_combine=pd.concat([df_metrics_dict,predicted_atoms_dict],axis=1)
                df_metrics_dict_single=pd.DataFrame(pd.concat([df_metrics_dict_single,df_metrics_dict_combine],ignore_index=True))


    loss = running_loss / len(loader)
    acc = 100 * correct / total
    return loss, acc,df_metrics_dict_single,df_predicted_atoms



head='7_7_50k'
attn='del'
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sample_file_name= f'data/result/statistics_supervision/uspto_yang/shield/ruizhen_product/{head}/npy/deep_attn_{attn}_{head}'
sample_file=os.path.join(parent_dir,sample_file_name)
label_file_name = f'data/result/statistics_supervision/uspto_yang/shield/ruizhen_product/{head}/label'
label_folder = os.path.join(parent_dir, label_file_name)
label_excel = os.path.join(label_folder, 'uspto_ruizhen_product_deep.xlsx')

# 参数配置
npy_dir = sample_file  # NPY文件目录
excel_path = label_excel  # Excel文件路径
smiles_col = "reactant"  # SMILES列名
label_col = "reactive_atoms_deep"  # 标签列名
max_length = 512  # 统一长度
max_atom=4

##para
network="transformer"
batch_size =1
epochs = 50
learn_rate = 5e-05
dropout=0.3
weight_decay=0
best_val_mcc=0.72
##save_files
save_name="network"
print("para:",network,batch_size,epochs,learn_rate,dropout,weight_decay)

custom_dataset = SmilesNpyDataset(
    npy_dir=npy_dir,
    excel_path=excel_path,
    smiles_column=smiles_col,
    label_column=label_col,
    max_length=max_length
)


# # A,B,C,D=custom_dataset[0]
# # # 固定随机种子，保证可重复
# torch.manual_seed(42)
# total_size = len(custom_dataset)
# # 2. 设置比例（可自定义）
# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1
# # 3. 计算各部分大小
# train_size = int(total_size * train_ratio)
# val_size = int(total_size * val_ratio)
# test_size = total_size - train_size - val_size
# # 4. 划分数据
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)


val_loader=DataLoader(custom_dataset, batch_size=batch_size)
result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/ruizhen_product/{head}/result/{save_name}'
result_folder = os.path.join(parent_dir, result_file_name)
para_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_'
           f'{weight_decay}_head_{head}_attn_{attn}_best_val_mcc_{best_val_mcc}.pth')
para_file=os.path.join(result_folder,para_path)

model = SmilesModel(input_size=max_length, output_size=max_length)
model.load_state_dict(torch.load(para_file,map_location='cpu'))

# 训练模型
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = 'cpu'
val_loss, val_acc,df_metrics_dict_single,df_predicted_atoms= evaluate_model(model, val_loader, device, max_atom)
path_excel_metrics=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_'
                    f'{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_mcc_{best_val_mcc}_metrics.xlsx')
para_file_excel_metrics=os.path.join(result_folder,path_excel_metrics)
df_metrics_dict_single.to_excel(para_file_excel_metrics)

path_excel_atoms=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_'
                  f'{weight_decay}_head_{head}_attn_{attn}_best_val_mcc_{best_val_mcc}_atoms.xlsx')
para_file_excel_atoms=os.path.join(result_folder,path_excel_atoms)
df_predicted_atoms.to_excel(para_file_excel_atoms)



















