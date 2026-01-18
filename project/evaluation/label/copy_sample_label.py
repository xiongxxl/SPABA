
from copy_flies_by_path import copy_and_rename_files
import numpy as np
import pandas as pd
import os


head='4_5'
##create new files
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
folder_npy_img= os.path.join(parent_dir, 'data/result/img_supervision/multi_1000_test/find_heads')

dest_file_name = f'data/deep_sample/{head}/npy'
dest_folder = os.path.join(parent_dir, dest_file_name)

label_file_name = f'data/deep_sample/{head}/label'
label_excel = os.path.join(parent_dir, label_file_name)

df_atoms=pd.read_excel(os.path.join(label_excel, 'double_criterion_10.xlsx'))
df_atoms_smiles=df_atoms['smiles']

for smiles in df_atoms_smiles:
    smiles_file_name=f'{smiles}_0.99'
    src_folder_npy=os.path.join(folder_npy_img,smiles_file_name)

    file_names_npy=f'{head}.npy'
    new_names_npy=f'{smiles}.npy'
    copy_and_rename_files(src_folder_npy, dest_folder, file_names_npy, new_names_npy)

    # file_names_png=f'{head}.png'
    # new_names_png=f'{smiles}_{head}.png'
    # copy_and_rename_files(src_folder_npy, dest_folder, file_names_png, new_names_png)
    #
    # ## copy img result
    # file_names_jpeg=f'{head}.jpeg'
    # new_names_jpeg=f'{smiles}_{head}.jpeg'
    # copy_and_rename_files(src_folder_npy, dest_folder, file_names_jpeg, new_names_jpeg)
