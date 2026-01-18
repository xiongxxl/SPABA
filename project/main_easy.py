import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from data_loader import SmilesNpyDataset
from model import SmilesModel
from train import train_model
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split

head='7_7'
attn='del'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sample_file_name= f'data/result/statistics_supervision/uspto_yang/shield/{head}/npy/deep_attn_{attn}_{head}'
sample_file=os.path.join(parent_dir,sample_file_name)
label_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/label'
label_folder = os.path.join(parent_dir, label_file_name)
label_excel = os.path.join(label_folder, 'uspto_yang_atom_by_template_reactant_index_v02_deep.xlsx')

# 参数配置
npy_dir = sample_file  # NPY文件目录
excel_path = label_excel  # Excel文件路径
smiles_col = "reactant"  # SMILES列名
label_col = "reactive_atoms_deep"  # 标签列名
max_length = 512  # 统一长度
max_atom=4

##para

network="2CNN+2FC"
batch_size = 1
epochs = 200
learn_rate = 5e-05
dropout=0.3
weight_decay=0

##save_files
save_name="network"

print("para:",head,network,batch_size,epochs,learn_rate,dropout,weight_decay)
custom_dataset = SmilesNpyDataset(
    npy_dir=npy_dir,
    excel_path=excel_path,
    smiles_column=smiles_col,
    label_column=label_col,
    max_length=max_length
)
# A,B,C,D=custom_dataset[0]

#划分数据集
torch.manual_seed(42)
train_size=int(len(custom_dataset)*0.8)
val_size=len(custom_dataset)-train_size
train_dataset,val_dataset=torch.utils.data.random_split(custom_dataset,[train_size,val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
# 固定随机种子，保证可重复
# torch.manual_seed(42)
# total_size = len(custom_dataset)
# # 2. 设置比例（可自定义）
# train_ratio = 0.8
# val_ratio = 0.2
# test_ratio = 0
#
# # 3. 计算各部分大小
# train_size = int(total_size * train_ratio)
# val_size = int(total_size * val_ratio)
# test_size = total_size - train_size - val_size
#
# # 4. 划分数据
# train_dataset, val_dataset, test_dataset = random_split(custom_dataset, [train_size, val_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmilesModel(input_size=max_length, output_size=max_length)

# if torch.cuda.device_count() > 1:
#     print("使用", torch.cuda.device_count(), "块 GPU 进行训练")
#     model = nn.DataParallel(model)  # 多GPU封装
# model.to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss(reduction='none')
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate,weight_decay=weight_decay)


# 训练模型

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
# trained_model, train_losses, val_losses, train_accs, val_accs, test_loss, test_acc = train_model(
#     model,
#     train_loader,
#     val_loader,
#     test_loader,
#     criterion,
#     optimizer,
#     num_epochs=epochs,
#     device=device,
#     max_atom= max_atom,
#
# )


best_model_state,best_val_acc, train_losses, val_losses, train_accs, val_accs= train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=epochs,
    device=device,
    max_atom= max_atom,
)


# Plot training curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss Curve')
plt.xlabel("Times ", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel("Times", fontsize=14)
plt.ylabel("ACC(%)", fontsize=14)
plt.legend()
plt.tight_layout()


result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
result_folder = os.path.join(parent_dir, result_file_name)
img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}.jpg'
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
# plt.show()


plt.figure(figsize=(6, 5))  # 只保留一个子图，可调整大小
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel("Times", fontsize=14)
plt.ylabel("ACC(%)", fontsize=14)
plt.legend()
plt.tight_layout()


result_folder = os.path.join(parent_dir, result_file_name)
img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}_only_ACC.jpg'
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
# plt.show()


para_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}.pth'
para_file=os.path.join(result_folder,para_path)
# 保存最终模型
torch.save(best_model_state, para_file)