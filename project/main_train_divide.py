import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import SmilesNpyDataset
from model import SmilesModel
from train import train_model
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick


head='divide_yang_50k'
attn='del'
# way='maximum'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sample_file_name= f'data/result/statistics_supervision/uspto_yang/shield/{head}/npy/train/deep_attn_{attn}_{head}_ruizhen_add'
sample_file_train=os.path.join(parent_dir,sample_file_name)
label_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/label/train'
label_folder = os.path.join(parent_dir, label_file_name)
label_excel_train = os.path.join(label_folder, 'uspto_yang_transformer_7_7_50k_train_split_ruizhen_add.xlsx')


sample_file_name= f'data/result/statistics_supervision/uspto_yang/shield/{head}/npy/test/deep_attn_{attn}_{head}'
sample_file_test=os.path.join(parent_dir,sample_file_name)
label_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/label/test'
label_folder = os.path.join(parent_dir, label_file_name)
label_excel_test = os.path.join(label_folder, 'uspto_yang_transformer_7_7_50k_test_split.xlsx')

sample_file_name= f'data/result/statistics_supervision/uspto_yang/shield/{head}/npy/valid/deep_attn_{attn}_{head}'
sample_file_valid=os.path.join(parent_dir,sample_file_name)
label_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/label/valid'
label_folder = os.path.join(parent_dir, label_file_name)
label_excel_valid = os.path.join(label_folder, 'uspto_yang_transformer_7_7_50k_val_split.xlsx')


# 参数配置
npy_dir_train = sample_file_train  # NPY文件目录
excel_path_train = label_excel_train  # Excel文件路径

npy_dir_valid = sample_file_valid  # NPY文件目录
excel_path_valid = label_excel_valid  # Excel文件路径

npy_dir_test = sample_file_test  # NPY文件目录
excel_path_test = label_excel_test  # Excel文件路径


smiles_col = "reactant"  # SMILES列名
label_col = "reactive_atoms_deep"  # 标签列名
max_length = 512  # 统一长度
max_atom=4

##para

network="transformer"
batch_size = 16
epochs = 50
learn_rate = 5e-05
drop=0.30
weight_decay=0
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
##save_files
save_name=("network")
datset="uspto_yang"
print("save_name",save_name)

print("para:",head,network, batch_size,epochs,learn_rate,drop,weight_decay)
custom_dataset_train = SmilesNpyDataset(
    npy_dir=npy_dir_train,
    excel_path=excel_path_train,
    smiles_column=smiles_col,
    label_column=label_col,
    max_length=max_length
)

custom_dataset_test = SmilesNpyDataset(
    npy_dir=npy_dir_test,
    excel_path=excel_path_test,
    smiles_column=smiles_col,
    label_column=label_col,
    max_length=max_length
)

custom_dataset_valid = SmilesNpyDataset(
    npy_dir=npy_dir_valid,
    excel_path=excel_path_valid,
    smiles_column=smiles_col,
    label_column=label_col,
    max_length=max_length
)





# A,B,C,D=custom_dataset[0]

#
# # 固定随机种子，保证可重复
# torch.manual_seed(42)
# total_size = len(custom_dataset)
# # 2. 设置比例（可自定义）
# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1
#
# # 3. 计算各部分大小
# train_size = int(total_size * train_ratio)
# val_size = int(total_size * val_ratio)
# test_size = total_size - train_size - val_size
# #
# # 4. 划分数据
# train_dataset, val_dataset, test_dataset = random_split(custom_dataset, [train_size, val_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_loader=DataLoader(custom_dataset_train, batch_size=batch_size)
test_loader=DataLoader(custom_dataset_test, batch_size=batch_size)
val_loader=DataLoader(custom_dataset_valid, batch_size=batch_size)


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

(best_model_state_dacc, best_model_state_mcc,best_model_state_sacc, best_val_dacc,best_val_mcc,best_val_sacc, train_losses,
val_losses,train_daccs, val_daccs, train_saccs, val_saccs, train_mccs, val_mccs,test_loss, test_dacc, test_sacc, test_mcc) = train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs=epochs,
    device=device,
    max_atom= max_atom,

)


# training curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train',linewidth=2.5)
plt.plot(val_losses, label='Validation',linewidth=2.5)
plt.title('Loss Curve',fontsize=24, fontname='Arial')
plt.xlabel('Times',fontsize=22, fontname='Arial')
plt.ylabel('Loss',fontsize=22, fontname='Arial')
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')

plt.legend()
legend=plt.legend()
legend.get_texts()[0].set_fontsize(14)
legend.get_texts()[1].set_fontsize(14)

plt.tight_layout()
result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
result_folder = os.path.join(parent_dir, result_file_name)
img_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
          f'{weight_decay}_head_{head}_attn_{attn}_best_val_dacc_{best_val_dacc}_loss_ruizhen_add.jpg')
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
# plt.show()

# ##中文标题
# plt.plot(train_losses, label='训练')
# plt.plot(val_losses, label='验证')
# plt.title('训练过程loss的曲线',fontsize=24, fontname='Arial')
# plt.xlabel('训练次数',fontsize=22, fontname='Arial')
# plt.ylabel('Loss',fontsize=22, fontname='Arial')
# plt.xticks(fontsize=18, fontname='Arial')
# plt.yticks(fontsize=18, fontname='Arial')
# plt.legend(['训练', '验证'],prop={'family': 'Arial', 'size': 16})
# plt.tight_layout()
# result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
# result_folder = os.path.join(parent_dir, result_file_name)
# img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_
# {weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}_loss_chinese.jpg'
# sample_file=os.path.join(result_folder,img_path)
# plt.savefig(sample_file)


## MCC
plt.figure(figsize=(8, 6))
plt.plot( train_mccs, label='Train',linewidth=2.5)
plt.plot(val_mccs, label='Validation',linewidth=2.5)
plt.title('MCC Curve',fontsize=24, fontname='Arial')
plt.xlabel('Times',fontsize=22, fontname='Arial')
plt.ylabel('MCC',fontsize=22, fontname='Arial')
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')


legend=plt.legend()
legend.get_texts()[0].set_fontsize(14)
legend.get_texts()[1].set_fontsize(14)
plt.tight_layout()
result_folder = os.path.join(parent_dir, result_file_name)
img_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
          f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_mcc}_MCC_ruizhen_add.jpg')
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
para_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
           f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_mcc}_MCC_ruizhen_add.pth')
para_file=os.path.join(result_folder,para_path)
torch.save(best_model_state_mcc, para_file)

# ##中文标题
# # plt.show()
# plt.plot(train_mccs, label='训练')
# plt.plot(val_mccs, label='验证')
# plt.title('训练过程MCC曲线',fontsize=24, fontname='Arial')
# plt.xlabel('训练次数',fontsize=22, fontname='Arial')
# plt.ylabel('MCC',fontsize=22, fontname='Arial')
# plt.xticks(fontsize=18, fontname='Arial')
# plt.yticks(fontsize=18, fontname='Arial')
# plt.legend(['训练', '验证'],prop={'family': 'Arial', 'size': 16})
# plt.tight_layout()
# result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
# result_folder = os.path.join(parent_dir, result_file_name)
# img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_
# weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}_MCC_chinese.jpg'
# sample_file=os.path.join(result_folder,img_path)
# plt.savefig(sample_file)


###DACC
plt.figure(figsize=(8, 6))  # 只保留一个子图，可调整大小
plt.plot(train_daccs, label='Train',linewidth=2.5)
plt.plot(val_daccs, label='Validation',linewidth=2.5)
plt.title('Defined Accuracy Curve',fontsize=24, fontname='Arial')
plt.xlabel('Times',fontsize=22, fontname='Arial')
plt.ylabel('DACC(%)',fontsize=22, fontname='Arial')
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')

legend=plt.legend()
legend.get_texts()[0].set_fontsize(14)
legend.get_texts()[1].set_fontsize(14)
plt.tight_layout()
result_folder = os.path.join(parent_dir, result_file_name)
img_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
          f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_dacc}_DACC_ruizhen_add.jpg')
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
para_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
           f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_dacc}_DACC_ruizhen_add.pth')
para_file=os.path.join(result_folder,para_path)
torch.save(best_model_state_dacc, para_file)

# ##中文曲线
# plt.plot(train_accs, label='训练')
# plt.plot(val_accs, label='验证')
# plt.title('训练过程曲线',fontsize=24, fontname='Arial')
# plt.xlabel('训练次数',fontsize=22, fontname='Arial')
# plt.ylabel('准确率',fontsize=22, fontname='Arial')
# plt.xticks(fontsize=18, fontname='Arial')
# plt.yticks(fontsize=18, fontname='Arial')
# plt.legend(['训练', '验证'],prop={'family': 'Arial', 'size': 16})
# plt.tight_layout()
# result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
# result_folder = os.path.join(parent_dir, result_file_name)
# img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_
# {weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}_ACC_chinese.jpg'
# sample_file=os.path.join(result_folder,img_path)
# plt.savefig(sample_file)

## SACC
plt.figure(figsize=(8, 6))
plt.plot(train_saccs, label='Train',linewidth=2.5)
plt.plot(val_saccs, label='Validation',linewidth=2.5)
plt.title(' Standard Accuracy Curve',fontsize=24, fontname='Arial')
plt.xlabel('Times',fontsize=22, fontname='Arial')
plt.ylabel('SACC(%)',fontsize=22, fontname='Arial')
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))

legend=plt.legend()
legend.get_texts()[0].set_fontsize(14)
legend.get_texts()[1].set_fontsize(14)
result_folder = os.path.join(parent_dir, result_file_name)
img_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
          f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_sacc}_SACC_ruizhen_add.jpg')
sample_file=os.path.join(result_folder,img_path)
plt.savefig(sample_file)
para_path=(f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{drop}_weight_delay_'
           f'{weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_sacc}_SACC_ruizhen_add.pth')
para_file=os.path.join(result_folder,para_path)
torch.save(best_model_state_sacc, para_file)

# ##中文曲线
# plt.plot(train_caccs, label='训练')
# plt.plot(val_caccs, label='验证')
# plt.title('训练过程曲线',fontsize=24, fontname='Arial')
# plt.xlabel('训练次数',fontsize=22, fontname='Arial')
# plt.ylabel('准确率',fontsize=22, fontname='Arial')
# plt.xticks(fontsize=18, fontname='Arial')
# plt.yticks(fontsize=18, fontname='Arial')
# plt.legend(['训练', '验证'],prop={'family': 'Arial', 'size': 16})
# plt.tight_layout()
# result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/{head}/result/{save_name}'
# result_folder = os.path.join(parent_dir, result_file_name)
# img_path=f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_{dropout}_weight_delay_
# {weight_decay}_head_{head}_attn_{attn}_best_val_acc_{best_val_acc}_SACC_chinese.jpg'
# sample_file=os.path.join(result_folder,img_path)
# plt.savefig(sample_file)
