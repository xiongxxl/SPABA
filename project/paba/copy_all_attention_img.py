import os
import sys

from copy_flies_by_path import copy_and_rename_files
import numpy as np
import pandas as pd


current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_reactive/double_para/combine')
df_criterion = pd.read_excel(os.path.join(input_files_smiles, 'double_criterion_100.xlsx'))
smiles=df_criterion['smiles']

data_files_img = 'data/result/img_reactive/double_para'
folder_name_attn = os.path.join(parent_dir, data_files_img)
folder_name_attn_npy = os.path.join(folder_name_attn, '64_heads')
folder_name_attn_combine = os.path.join(folder_name_attn, 'combine')

head_list=['4_5','7_7']
for head in head_list:
    for smile in smiles:
        smiles_file_name = f'{smile}_0.99'
        src_folder_npy = os.path.join(folder_name_attn_npy, smiles_file_name)
        file_names_png = f'{head}.png'
        new_names_png = f'{smile}_{head}.png'
        dest_folder = folder_name_attn_combine
        copy_and_rename_files(src_folder_npy, dest_folder, file_names_png, new_names_png)



