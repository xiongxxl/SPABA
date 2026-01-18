import matplotlib.pyplot as plt
import os
import pandas as pd
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

df_head_4_5=pd.DataFrame()
df_head_7_7=pd.DataFrame()
df_head_combine=pd.DataFrame()

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
##4_5 head
file_name='data/result/statistics_reactive/double_para/find_alpha/df_ratio_multi_0.97_0.99_4_5.xlsx'
input_files_smiles = os.path.join(parent_dir, file_name)
df_ratio=pd.read_excel(input_files_smiles)

df_head_4_5['ratio']=df_ratio['ratio']
df_head_4_5['first_dict'] = df_ratio['head_rank'].apply(extract_first_dict)
df_head_4_5['heads'] = df_head_4_5['first_dict'].str.split(':').str[0]
df_head_4_5['number'] = df_head_4_5['first_dict'].str.split(':').str[1].str.replace(' ','')
df_head_4_5 = df_head_4_5.drop(columns=['first_dict'])
excel_multi_4_5='data/result/statistics_reactive/double_para/find_alpha/df_ratio_multi_first_4_5.xlsx'
filename_detail_4_5 = os.path.join(parent_dir, excel_multi_4_5)
df_head_4_5.to_excel(filename_detail_4_5)

## 7_7 head
file_name='data/result/statistics_reactive/double_para/find_alpha/df_ratio_multi_0.97_0.99_7_7.xlsx'
input_files_smiles = os.path.join(parent_dir, file_name)
df_ratio=pd.read_excel(input_files_smiles)

df_head_7_7['ratio']=df_ratio['ratio']
df_head_7_7['first_dict'] = df_ratio['head_rank'].apply(extract_first_dict)
df_head_7_7['heads'] = df_head_7_7['first_dict'].str.split(':').str[0]
df_head_7_7['number'] = df_head_7_7['first_dict'].str.split(':').str[1].str.replace(' ','')
df_head_7_7 = df_head_7_7.drop(columns=['first_dict'])
excel_multi_first='data/result/statistics_reactive/double_para/find_alpha/df_ratio_multi_first_7_7.xlsx'
filename_detail_7_7= os.path.join(parent_dir, excel_multi_first)
df_head_7_7.to_excel(filename_detail_7_7)

df_head_combine['ratio']=df_ratio['ratio']
df_head_4_5['number']=pd.to_numeric(df_head_4_5['number'],errors='coerce')
df_head_7_7['number']=pd.to_numeric(df_head_7_7['number'],errors='coerce')
df_head_combine['number']=df_head_4_5['number']+df_head_7_7['number']
excel_multi_first='data/result/statistics_reactive/double_para/find_alpha/df_ratio_multi_first_combine.xlsx'
filename_detail_combine= os.path.join(parent_dir, excel_multi_first)
df_head_combine.to_excel(filename_detail_combine)

# ##show results ###
ratio_4_5=df_head_4_5['ratio']
number_4_5=df_head_4_5['number']
ratio_7_7=df_head_7_7['ratio']
number_7_7=df_head_7_7['number']
ratio_combine=df_head_combine['ratio']
number_combine=df_head_combine['number']

plt.figure(figsize=(10,8))
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio_4_5,number_4_5, marker='o', linestyle='-', color='#32CD32',label='4_5')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio_7_7,number_7_7, marker='^', linestyle='-', color='#800080',label='7_7')
# plt.scatter(df_head_first['ratio'],df_head_first['number'], label='Data Points', c='blue', alpha=0.7)
plt.plot(ratio_combine,number_combine, marker='s', linestyle='-', color='#FF6347',label='sum')
#FDC897
plt.legend(loc='upper left')
plt.title("Result of 4_5 7_7 and sum", fontsize=16, fontweight='bold')
plt.xlabel("Alpha values ", fontsize=14, fontweight='bold')
plt.ylabel("Number", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# plt.tight_layout()
data_files_smiles='data/result/statistics_reactive/double_para/find_alpha'
ratio_img=os.path.join(parent_dir, data_files_smiles)
file_name=f'sum.jpeg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail,dpi=300)
plt.show()