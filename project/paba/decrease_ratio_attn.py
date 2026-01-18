import numpy as np
import os
from tokenizer import tokenize_smiles

from remove_letter_number import find_and_remove_non_letters

import pandas as pd

from restrain_functional_reactive import restrain_functional_axis

def remove_rows_and_columns(matrix, rows_to_remove, cols_to_remove):
    # 转换输入列表为NumPy数组
    matrix_np = np.array(matrix)
    # 删除指定的行
    matrix_np = np.delete(matrix_np, rows_to_remove, axis=0)
    # 删除指定的列
    matrix_np = np.delete(matrix_np, cols_to_remove, axis=1)
    return matrix_np

#folder_path = './result/attention/first/img/CCCCCCCCCCCCCCO'
# 遍历根目录下的所有文件夹
#df_frag_smiles = pd.DataFrame(columns=["simles","frag","functional_group","location","attention_axis"])


def decrease_ratio_highlight_atoms(folder_path,filename_without_extension):

    attn_binary = np.load(folder_path)
    #np.fill_diagonal(attn_binary,0) #set diagonal element to 0
    attn_binary_del=attn_binary
    # del non-character
    filename_without_extension_re = tokenize_smiles(filename_without_extension)
    ## delete alpha
    #positions, cleaned_string = find_non_alphanumeric_positions_and_remove(filename_without_extension_re)
    positions, cleaned_string = find_and_remove_non_letters(filename_without_extension_re)
    # np.save(f'{filename_without_extension}c
    # print('cleaned_string is:',cleaned_string)
    # print('positions is:', positions)
    # print('attn_binary :', np.shape(attn_binary))
    # print(attn_binary_del)

    # del Delete the corresponding rows and columns
    attn_binary_letter = remove_rows_and_columns(attn_binary_del, positions, positions)
    #print('attn_binary_del:', np.shape(attn_binary_letter))

    # find no_zero_element
    non_zero_positions = np.nonzero(attn_binary_letter)
    non_zero_col=np.unique(non_zero_positions[0]).tolist()
    ##concatatenate nadarry  and del copy
    # non_zero_positions_combined = np.concatenate(non_zero_positions[0], non_zero_positions[1])
    # non_zero_positions_combined_unique = np.unique(non_zero_positions_combined)

    #non_zero_row=np.unique(non_zero_positions[1]).tolist()
    # print("不为零元素的行索引：", non_zero_col)
    # print("不为零元素的列索引：", non_zero_row)

    ## find functonal group
    smiles = filename_without_extension
    atom_indices = non_zero_col

    # frag_indices_del_benzene = remove_benzene_indices(smiles,non_zero_col)
    # atom_indices=frag_indices_del_benzene
    results,reactive_atoms_restrain= restrain_functional_axis(smiles, atom_indices)

    return reactive_atoms_restrain