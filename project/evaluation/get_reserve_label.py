#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


############################
# 1. 定义 SmilesNpyDataset #
############################

class SmilesNpyDataset(Dataset):
    def __init__(self, npy_dir, excel_path, smiles_column, label_column, max_length):
        """
        参数:
            npy_dir: 包含 NPY 文件的目录
            excel_path: 包含标签的 Excel 文件路径
            smiles_column: Excel 中 SMILES 字符串的列名（如 "reactant"）
            label_column: Excel 中标签列名（如 "reactive_atoms_deep"）
            max_length: 统一处理后的最大长度
        """
        self.max_length = max_length
        self.npy_files = list(Path(npy_dir).glob("*.npy"))  # 所有 npy 文件
        self.df = pd.read_excel(excel_path)                 # 原始 Excel 数据

        # 建立 smiles -> npy 文件路径 映射，假设 npy 文件名 = SMILES 字符串
        self.smiles_to_file = {f.stem: f for f in self.npy_files}

        self.samples = []          # (smiles, data(ndarray), label(ndarray))
        used_row_indices = []      # 记录 Excel 中真正有 npy 文件的行

        # 遍历 Excel，每一行如果在 npy 映射中，就加入 samples
        missing_files = 0
        for idx, row in self.df.iterrows():
            smiles = row[smiles_column]
            if smiles in self.smiles_to_file:
                npy_path = self.smiles_to_file[smiles]
                data = np.load(npy_path)
                label = np.array(eval(row[label_column]))  # Excel 中是字符串形式的 list

                self.samples.append((smiles, data, label))
                used_row_indices.append(idx)
            else:
                missing_files += 1

        print(f"总行数: {len(self.df)}, 成功匹配 NPY 的样本数: {len(self.samples)}, 缺失 NPY 文件的行数: {missing_files}")

        # 只保留真正参与训练的那些行，对齐到 samples 的顺序
        self.filtered_df = self.df.iloc[used_row_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smiles, data, label = self.samples[idx]

        # 统一处理 data、label 长度（与训练保持一致）
        data = self._pad_or_truncate(data)
        label = self._pad_or_truncate(label)
        return torch.FloatTensor(data), torch.FloatTensor(label.reshape(-1)), len(smiles), smiles

    def _pad_or_truncate(self, arr):
        """将数组填充或截断到 max_length"""
        if len(arr.shape) == 1:
            current_len = arr.shape[0]
            if current_len < self.max_length:
                pad_width = self.max_length - current_len
                arr = np.pad(arr, (0, pad_width), "constant")
            elif current_len > self.max_length:
                arr = arr[:self.max_length]
            return arr

        elif len(arr.shape) == 2:
            rows, cols = arr.shape
            # 行、列都补到 max_length（和你原逻辑尽量保持一致）
            if rows < self.max_length or cols < self.max_length:
                pad_row = max(0, self.max_length - rows)
                pad_col = max(0, self.max_length - cols)
                arr = np.pad(arr, ((0, pad_row), (0, pad_col)), "constant")
            if arr.shape[0] > self.max_length or arr.shape[1] > self.max_length:
                arr = arr[:self.max_length, :self.max_length]
            return arr

        else:
            raise ValueError("不支持 3 维以上数组")


#########################################
# 2. 划分 train / val / test 并导出 Excel #
#########################################

def export_split(full_dataset: SmilesNpyDataset, split_subset, filename: str):
    """
    根据 random_split 的 Subset，将对应行从 filtered_df 中导出到 Excel

    full_dataset: 原始 SmilesNpyDataset（包含 filtered_df）
    split_subset: random_split 得到的 Subset（train_dataset / val_dataset / test_dataset）
    filename: 输出文件名
    """
    df = full_dataset.filtered_df  # 只包含真正有 npy 文件的行
    indices = split_subset.indices

    split_df = df.iloc[indices].copy()
    split_df.to_excel(filename, index=False)
    print("已保存:", filename)


##########################
# 3. 主程序入口 main()    #
##########################

def main():
    # ==== 你原来的参数部分 ====
    head = "7_7_50k"
    attn = "del"

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    # NPY 目录
    sample_file_name = f"data/result/statistics_supervision/uspto_yang/shield/{head}/npy/deep_attn_{attn}_{head}"
    npy_dir = os.path.join(parent_dir, sample_file_name)

    # Excel 标签文件
    label_file_name = f"data/result/statistics_supervision/uspto_yang/shield/{head}/label"
    label_folder = os.path.join(parent_dir, label_file_name)
    excel_path = os.path.join(label_folder, "uspto_yang_reactive_atom_orgin_50k_indices_atoms_deep.xlsx")

    smiles_col = "reactant"           # SMILES 列名
    label_col = "reactive_atoms_deep" # 标签列名
    max_length = 512                  # 统一长度

    # 一些训练参数（这里只是打印，方便你核对）
    network = "transformer"
    batch_size = 1
    epochs = 100
    learn_rate = 5e-05
    dropout = 0.3
    weight_decay = 0
    best_val_mcc = 0.71

    print("para:", network, batch_size, epochs, learn_rate, dropout, weight_decay)
    print("NPY 目录:", npy_dir)
    print("Excel 标签路径:", excel_path)

    # ==== 构建 Dataset ====
    custom_dataset = SmilesNpyDataset(
        npy_dir=npy_dir,
        excel_path=excel_path,
        smiles_column=smiles_col,
        label_column=label_col,
        max_length=max_length
    )

    # 固定随机种子以保证划分可复现
    torch.manual_seed(42)

    total_size = len(custom_dataset)
    print("总样本数量(可用于训练):", total_size)

    # 数据集划分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 计算各部分大小
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    print("train_size:", train_size, "val_size:", val_size, "test_size:", test_size)

    # random_split
    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset,
        [train_size, val_size, test_size]
    )

    # ==== 导出 3 个 Excel ====
    export_split(custom_dataset, train_dataset, "network_transformer_train_split.xlsx")
    export_split(custom_dataset, val_dataset,   "network_transformer_val_split.xlsx")
    export_split(custom_dataset, test_dataset,  "network_transformer_test_split.xlsx")


if __name__ == "__main__":
    main()

