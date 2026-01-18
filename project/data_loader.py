import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tokenizer import tokenize_smiles

class SmilesNpyDataset(Dataset):
    def __init__(self, npy_dir, excel_path, smiles_column, label_column, max_length):

        self.max_length = max_length
        self.npy_files = list(Path(npy_dir).glob('*.npy'))
        self.df = pd.read_excel(excel_path)

        # 创建SMILES到文件路径的映射        """
        #         参数:
        #             npy_dir: 包含NPY文件的目录
        #             excel_path: 包含标签的Excel文件路径
        #             smiles_column: Excel中SMILES字符串的列名
        #             label_column: Excel中标签的列名
        #             max_length: 统一处理后的最大长度
        # temp_dir = 'temp_mol_imgs'
        # os.makedirs(temp_dir, exist_ok=True)
        self.smiles_to_file = {f.stem: f for f in self.npy_files}
        C=self.smiles_to_file
        # 验证数据
        self.samples = []
        missing_files = 0

        for _, row in self.df.iterrows():
            smiles = row[smiles_column]
            if smiles in self.smiles_to_file:
                data = np.load(self.smiles_to_file[smiles])
                label = np.array(eval(row[label_column]))
                self.samples.append((smiles, data, label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smiles, data, label = self.samples[idx]

        # D=data.shape[0]
        # 统一处理数据到max_length
        data = self._pad_or_truncate(data)
        # 处理标签
        label = self._pad_or_truncate(label)  # 确保标签是1D
        return torch.FloatTensor(data), torch.FloatTensor(label.reshape(-1)), len(tokenize_smiles(smiles)),smiles

    def _pad_or_truncate(self, arr):
        """将数组填充或截断到max_length"""
        if len(arr.shape) == 1:
            current_len = arr.shape[0]
            if current_len < self.max_length:
                pad_width = self.max_length - current_len
                arr = np.pad(arr, (0, pad_width), 'constant')
            elif current_len > self.max_length:
                arr = arr[:self.max_length]
            return arr
        elif len(arr.shape) == 2:

            if arr.shape[0] < self.max_length:
                pad_width_row =self.max_length - arr.shape[0]
                pad_width_col= self.max_length - arr.shape[1]
                arr = np.pad(arr, ((0, pad_width_row), (0, pad_width_col)), 'constant')

            elif arr.shape[0] > self.max_length:
                arr = arr[:self.max_length, :self.max_length]
            return arr
        else:
            raise ValueError("不支持3维以上数组")

# 使用示例
if __name__ == "__main__":
    head='all'
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    sample_file_name= f'paba/data/middle_attention/name_sample/{head}/npy/del_1000_v04'
    sample_file=os.path.join(parent_dir,sample_file_name)
    label_file_name = f'paba/data/middle_attention/name_sample/{head}/label/'
    label_folder = os.path.join(parent_dir, label_file_name)
    label_excel = os.path.join(label_folder, 'output_with_highlight_1500 _del_v02_20250423_deep_replace.xlsx')

    # 参数配置
    NPY_DIR = sample_file  # NPY文件目录
    EXCEL_PATH = label_excel  # Excel文件路径
    SMILES_COL = "smiles"  # SMILES列名
    LABEL_COL = "deep_criterion"  # 标签列名
    MAX_LENGTH = 512  # 统一长度
    BATCH_SIZE = 1
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # 加载数据
    dataset = SmilesNpyDataset(
        npy_dir=NPY_DIR,
        excel_path=EXCEL_PATH,
        smiles_column=SMILES_COL,
        label_column=LABEL_COL,
        max_length=MAX_LENGTH
    )
    #
    A,B,C=dataset[1]
    print(A)
    print(B)
    print(C)




    # class NpyExcelDataset(Dataset):
    #     def __init__(self, npy_files, excel_path, sheet_name, data_column, label_column, transform=None):
    #         """
    #         参数:
    #             npy_files: npy文件路径列表
    #             excel_path: Excel文件路径
    #             sheet_name: Excel工作表名
    #             data_column: Excel中标识npy文件名的列
    #             label_column: Excel中包含标签的列
    #             transform: 可选的数据转换函数
    #         """
    #         self.npy_files = npy_files
    #         self.transform = transform
    #
    #         # 加载Excel文件
    #         self.df = pd.read_excel(excel_path, sheet_name=sheet_name)
    #
    #         # 创建文件名到数据的映射
    #         self.data_dict = {}
    #         for file in npy_files:
    #             key = file.stem if hasattr(file, 'stem') else Path(file).stem
    #             self.data_dict[key] = np.load(file)
    #
    #         # 创建样本索引列表
    #         self.samples = []
    #         for idx, row in self.df.iterrows():
    #             file_key = row[data_column]
    #             if file_key in self.data_dict:
    #                 data = self.data_dict[file_key]
    #                 label = eval(row[label_column])
    #                 self.samples.append((file_key,idx, label))
    #                 # 如果npy文件包含多个样本，为每个样本添加条目
    #                 # if len(data.shape) > 1:
    #                 #     for i in range(data.shape[0]):
    #                 #         self.samples.append((file_key, i, label))
    #                 # else:
    #                 #     self.samples.append((file_key, 0, label))
    #
    #     def __len__(self):
    #         return len(self.samples)
    #
    #     def __getitem__(self, idx):
    #         file_key, sample_idx, label = self.samples[idx]
    #         data = self.data_dict[file_key]
    #         sample=data
    #         # 获取特定样本
    #         # if len(data.shape) > 1:
    #         #     sample = data[sample_idx]
    #         # else:
    #         #     sample = data
    #
    #         if self.transform:
    #             sample = self.transform(sample)
    #
    #         # 转换为torch张量
    #         sample_tensor = torch.from_numpy(sample).float()
    #         label_tensor = torch.tensor(label, dtype=torch.long)
    #
    #         return sample_tensor, label_tensor


    # head='4_5'
    # current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)
    # sample_file_name=f'data/deep_sample/{head}/npy'
    # sample_file=os.path.join(parent_dir,sample_file_name)
    # label_file_name = f'data/deep_sample/{head}/label'
    # label_folder = os.path.join(parent_dir, label_file_name)
    # label_excel = os.path.join(label_folder, 'double_criterion_10_deep.xlsx')
    # npy_dir = Path(sample_file)
    # npy_files = list(npy_dir.glob("*.npy"))
    #
    # # 2. 创建数据集
    # custom_dataset = NpyExcelDataset(
    #     npy_files=npy_files,
    #     excel_path=label_excel,
    #     sheet_name="Sheet1",
    #     data_column="smiles",  # Excel中标识npy文件名的列
    #     label_column="deep_criterion"  # Excel中包含标签的列
    # )
    #
    # # sample, label = custom_dataset[2]  # 这会调用 __getitem__
    # # print("Sample shape:", sample.shape)
    #
    # train_size=int(len(custom_dataset)*0.7)
    # val_size=len(custom_dataset)-train_size
    # train_dataset,val_dataset=torch.utils.data.random_split(custom_dataset,[train_size,val_size])
    #
    # # 4. 创建DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)


