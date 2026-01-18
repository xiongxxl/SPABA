import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
#4_5 head
input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_reactive/double_para/find_alpha')
excel_name='df_ratio_multi_0.97_0.99_4_5.xlsx'
fold_excel_path=os.path.join(input_files_smiles,excel_name)
df_ratio=pd.read_excel(fold_excel_path)

ratio=df_ratio['ratio']
atoms_error_0=df_ratio['0_number']
atoms_error_1=df_ratio['1_number']
atoms_error_2=df_ratio['2_number']
atoms_error_3=df_ratio['3_number']
atoms_error_4=df_ratio['4_number']

plt.figure(figsize=(10,8))
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_0, marker='o', linestyle='-', color='#32CD32',label='0_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_1, marker='^', linestyle='-', color='#800080',label='1_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_2, marker='s', linestyle='-', color='#FF6347',label='2_atoms_error')

plt.plot(ratio,atoms_error_3, marker='*', linestyle='-', color='#9DD79D',label='3_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_4, marker='+', linestyle='-', color='#9ABBF3',label='4_atoms_error')

#FDC897
plt.legend(loc='upper left')
plt.title("Atoms_error changes with different alpha value of 4_5 head ", fontsize=14, fontweight='bold')
plt.xlabel(" Alpha values ", fontsize=14, fontweight='bold')
plt.ylabel("Number of sample", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# plt.tight_layout()
data_files_smiles='data/result/statistics_reactive/double_para/find_alpha'
ratio_img=os.path.join(parent_dir, data_files_smiles)
file_name=f'Atoms_error changes with different 4_5.jpeg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail,dpi=300)
plt.show()

##7_7 head
excel_name='df_ratio_multi_0.97_0.99_7_7.xlsx'
fold_excel_path=os.path.join(input_files_smiles,excel_name)
df_ratio=pd.read_excel(fold_excel_path)

ratio=df_ratio['ratio']
atoms_error_0=df_ratio['0_number']
atoms_error_1=df_ratio['1_number']
atoms_error_2=df_ratio['2_number']
atoms_error_3=df_ratio['3_number']
atoms_error_4=df_ratio['4_number']


plt.figure(figsize=(10,8))
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_0, marker='o', linestyle='-', color='#32CD32',label='0_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_1, marker='^', linestyle='-', color='#800080',label='1_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_2, marker='s', linestyle='-', color='#FF6347',label='2_atoms_error')

plt.plot(ratio,atoms_error_3, marker='*', linestyle='-', color='#9DD79D',label='3_atoms_error')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,atoms_error_4, marker='+', linestyle='-', color='#9ABBF3',label='4_atoms_error')

#FDC897
plt.legend(loc='upper left')
plt.title("Atoms_error changes with different alpha value of 7_7 head ", fontsize=14, fontweight='bold')
plt.xlabel(" Alpha values ", fontsize=14, fontweight='bold')
plt.ylabel("Number of sample", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# plt.tight_layout()
data_files_smiles='data/result/statistics_reactive/double_para/find_alpha'
ratio_img=os.path.join(parent_dir, data_files_smiles)
file_name=f'Atoms_error changes with different 7_7.jpeg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail,dpi=300)
plt.show()


import matplotlib.pyplot as plt
# 数据
categories = ['0', '1', '2', '3', '4']
values = [7,8,12,6,5]
# 创建柱状图
plt.bar(categories, values, color='#b4e0f6')
# 添加标题和标签

plt.title("Atoms error in alpha 0.99 of 4_5 head", fontsize=12, fontweight='bold')
plt.xlabel("Atoms_error", fontsize=14, fontweight='bold')
plt.ylabel("Number of sample", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

file_name=f'Atoms error in alpha of 4_5.jpeg'
ratio_img_detail=os.path.join(input_files_smiles,file_name)
plt.savefig(ratio_img_detail)
plt.show()

import matplotlib.pyplot as plt
# 数据
categories = ['0', '1', '2', '3', '4']
values = [0,5,10,8,9]
# 创建柱状图
plt.bar(categories, values, color='#b4e0f6')
# 添加标题和标签

plt.title("Atoms error in alpha 0.99 of 7_7 head", fontsize=12, fontweight='bold')
plt.xlabel("Atoms_error", fontsize=14, fontweight='bold')
plt.ylabel("Number of sample", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

file_name=f'Atoms error in alpha of 7_7 head'
ratio_img_detail=os.path.join(input_files_smiles,file_name)
plt.savefig(ratio_img_detail)
plt.show()
