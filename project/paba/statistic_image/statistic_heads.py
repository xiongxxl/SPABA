import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = 'Arial'
import os
import pandas as pd
import ast
def extract_first_dict(row):
    if pd.isna(row):  # 检查是否为空
        return None
    # 按逗号分隔，提取第一个部分
    first_dict = row.split(",")[0].strip()
    return first_dict
# 应用函数到整列

def extract_second_dict(row):
    if pd.isna(row):  # 检查是否为空
        return None
    # 按逗号分隔，提取第一个部分
    second_dict = row.split(",")[1].strip()
    return second_dict
# 应用函数到整列

def extract_third_dict(row):
    if pd.isna(row):  # 检查是否为空
        return None
    # 按逗号分隔，提取第一个部分
    third_dict = row.split(",")[2].strip()
    return third_dict

df_head_first=pd.DataFrame()
df_head_second=pd.DataFrame()
df_head_third=pd.DataFrame()

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
file_name='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_0.90_0.99_double_v01_20250224.xlsx'
input_files_smiles = os.path.join(parent_dir, file_name)
df_ratio=pd.read_excel(input_files_smiles)

# top one
df_head_first['ratio']=df_ratio['ratio']
df_head_first['first_dict'] = df_ratio['head_rank'].apply(extract_first_dict)
df_head_first['heads'] = df_head_first['first_dict'].str.split(':').str[0]
df_head_first['number'] = df_head_first['first_dict'].str.split(':').str[1].str.replace(' ','')
df_head_first = df_head_first.drop(columns=['first_dict'])


ratio_multi_first='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_first.xlsx'
df_ratio_input_filename_first = os.path.join(parent_dir, ratio_multi_first)
df_head_first.to_excel(df_ratio_input_filename_first)

df_head_first_sample=df_head_first.iloc[::3,:]
ratio_multi_first_v01='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_first_v01.xlsx'
df_ratio_input_filename_first_v01 = os.path.join(parent_dir, ratio_multi_first_v01)
df_head_first_sample.to_excel(df_ratio_input_filename_first_v01)

# top two
df_head_second['ratio']=df_ratio['ratio']
df_head_second['first_dict'] = df_ratio['head_rank'].apply(extract_second_dict)
df_head_second['heads'] = df_head_second['first_dict'].str.split(':').str[0]
df_head_second['number'] = df_head_second['first_dict'].str.split(':').str[1].str.replace(' ','')
df_head_second = df_head_second.drop(columns=['first_dict'])
ratio_multi_second='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_second.xlsx'
df_ratio_input_filename_second= os.path.join(parent_dir, ratio_multi_second)
df_head_second.to_excel(df_ratio_input_filename_second)

df_head_second_sample=df_head_second.iloc[::3,:]
ratio_multi_second_v01='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_second_v01.xlsx'
df_ratio_input_filename_second_v01 = os.path.join(parent_dir, ratio_multi_second_v01)
df_head_second_sample.to_excel(df_ratio_input_filename_second_v01)

## top three
df_head_third['ratio']=df_ratio['ratio']
df_head_third['first_dict'] = df_ratio['head_rank'].apply(extract_third_dict)
df_head_third['heads'] = df_head_third['first_dict'].str.split(':').str[0]
df_head_third['number'] = df_head_third['first_dict'].str.split(':').str[1].str.replace(' ','')

df_head_third = df_head_third.drop(columns=['first_dict'])
ratio_multi_third='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_third.xlsx'
df_ratio_input_filename_third = os.path.join(parent_dir, ratio_multi_third)
df_head_third.to_excel(df_ratio_input_filename_third)

df_head_third_sample=df_head_third.iloc[::3,:]
ratio_multi_third_v01='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_third_v01.xlsx'
df_ratio_input_filename_third_v01 = os.path.join(parent_dir, ratio_multi_third_v01)
df_head_third_sample.to_excel(df_ratio_input_filename_third_v01)


##show results ###
ratio_multi_first_1='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_first_v01.xlsx'
filename_sample_first = os.path.join(parent_dir, ratio_multi_first_1)
df_head_first=pd.read_excel(filename_sample_first)
ratio=df_head_first['ratio']
number_first=pd.to_numeric(df_head_first['number'],errors='coerce')
heads_first=df_head_first['heads']

ratio_multi_second_1='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_second_v01.xlsx'
filename_sample_second= os.path.join(parent_dir, ratio_multi_second_1)
df_head_second=pd.read_excel(filename_sample_second)
number_second=pd.to_numeric(df_head_second['number'],errors='coerce')
heads_second=df_head_second['heads']

ratio_multi_third_1='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads/df_ratio_multi_third_v01.xlsx'
filename_sample_third= os.path.join(parent_dir, ratio_multi_third_1)
df_head_third=pd.read_excel(filename_sample_third)
number_third=pd.to_numeric(df_head_third['number'],errors='coerce')
heads_third=df_head_third['heads']

plt.figure(figsize=(10,8))
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,number_first, marker='o', linestyle='-', color='#32CD32',label='First')
for i, head in enumerate(heads_first):
    plt.text(ratio[i], number_first[i], f"{head}", fontsize=9, ha='right')


ratio=df_head_second['ratio']
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,number_second, marker='^', linestyle='-', color='#800080',label='Second')
for i, head in enumerate(heads_second):
    plt.text(ratio[i], number_second[i], f"{head}", fontsize=9, ha='right')

ratio=df_head_third['ratio']
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio,number_third, marker='s', linestyle='-', color='#FF6347',label='Third')
for i, head in enumerate(heads_third):
    plt.text(ratio[i], number_third[i], f"{head}", fontsize=9, ha='right')

plt.legend(loc='upper left')
plt.title("Number of top three heads in different alpha values ", fontsize=16, fontweight='bold')
plt.xlabel("Alpha values ", fontsize=14, fontweight='bold')
plt.ylabel("Number", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# plt.tight_layout()
data_files_smiles='data/result/statistics_supervision/multi_1000_double_20_atoms/find_heads'
ratio_img=os.path.join(parent_dir, data_files_smiles)
file_name=f'top three heads.jpeg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()  # 显示图形


