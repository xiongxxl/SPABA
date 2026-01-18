import pandas as pd
import numpy as np
import os
import re

def get_diff(A, B):
    A = sorted(A)
    B = sorted(B)
    diff = -1
    for b in B:
        if b in A:
            pass
        else:
            return diff
    diff = len(A) - len(B)
    return diff


# print(get_diff(A,B))


# def is_similar(arr1, arr2):
#     # 如果长度不同，直接返回False
#     # if len(arr1) != len(arr2):
#     #     return False
#     # 计算两数组中不同元素的数量
#     diff_count = sum([1 for x, y in zip(arr1, arr2) if x != y])
#     # 如果不同元素的数量为0或1，返回True
#     return diff_count

# def contains(sub_array, main_array):
#     return all(item in main_array for item in sub_array)
# # 示例
# # main_array = [1, 2, 3, 4, 5]
# # sub_array = [2, 3]
# # print(contains(sub_array, main_array))  # 输出 True
# # sub_array = [2, 6]
# # print(contains(sub_array, main_array))  # 输出 False
# def check_and_count_difference(A, B):
#     # 将 A 和 B 转换为集合
#     set_A = set(A)
#     set_B = set(B)
#
#     # 检查 A 是否包含 B
#     if set_B.issubset(set_A):
#         # 计算 A 和 B 的元素数量差
#         numbers_A = re.findall(r'\d+',A )
#         numbers_B = re.findall(r'\d+', B)
#         # 筛选出大于 10 的数字并计算数量
#         count_A = sum(1 for num in numbers_A)
#         count_B=sum(1 for num in numbers_B)
#         difference = count_A-count_B
#         return difference, True
#     else:
#         return None, False
#
#
# # 示例
# A = [1, 2, 3, 4, 5]
# B = [2, 5,3]
# erro, flag = check_and_count_difference(A, B)
# print("元素个数差为：", erro if flag else "A 不包含 B")


def calculate_reactive_percentage(df_criterion, df_sample):
    df_error_atoms = pd.DataFrame()
    df_criterion_axis = df_criterion[['smiles', 'gold_criterion']]

    tmp_location = {}
    tmp_attention_axis = {}
    tmp_flag={}
    for atom_num in range(6):
        atom_num = str(atom_num)

        for index_criterion, row_criterion in df_criterion_axis.iterrows():
            tmp_location[index_criterion] = []
            tmp_attention_axis[index_criterion] = []
            tmp_flag[index_criterion] = []

            value_smiles_criterion = row_criterion['smiles']
            value_location_criterion = row_criterion['gold_criterion']
            # 这里可以对value1和value2进行操作
            # print(value_smiles_criterion, value_location_criterion)
            filtered_df = df_sample[df_sample['smiles'] == value_smiles_criterion]
            filtered_df_64 = filtered_df.reset_index(drop=True)
            attention_axis_multi=[]
            # # 遍历DataFrame的每一行，与数组进行比较
            for index_sample, row_sample in filtered_df_64.iterrows():
                # if set(value_location_criterion).issubset(set(row_sample['location'])):
                # if contains(value_location_criterion,row_sample['location']):
                row_sample_location = row_sample['reactive_atoms']

                #A=row_sample['location']
                row_sample_location = eval(row_sample_location)
                value_location_criterion = eval(str(value_location_criterion))
                erro = get_diff(row_sample_location, value_location_criterion)
                if erro == eval(atom_num):
                    tmp_flag[index_criterion]=1
                    # df_criterion.at[index_criterion, column_name_location] = row_sample['location']
                    tmp_location[index_criterion].append(row_sample['reactive_atoms'])
                    # df_criterion.at[index_criterion, column_name_attention_axis] = row_sample['attention_axis']
                    tmp_attention_axis[index_criterion].append(row_sample['head_axis'])

        column_name_flag = f'flag_{atom_num}'
        column_name_location = f'reactive_atoms_{atom_num}'
        column_name_attention_axis = f'head_axis_{atom_num}'
        df_flag_tmp= pd.DataFrame(list(tmp_flag.items()), columns=['index', 'flag'])
        df_criterion[column_name_flag]=df_flag_tmp['flag']
        df_location_tmp= pd.DataFrame(list(tmp_location.items()), columns=['index', 'reactive_atoms'])
        df_criterion[column_name_location]=df_location_tmp['reactive_atoms']
        df_attention_axis_tmp = pd.DataFrame(list(tmp_attention_axis.items()), columns=['index', 'head_axis'])
        df_criterion[column_name_attention_axis] = df_attention_axis_tmp['head_axis']


    return df_criterion

if __name__ == "__main__":
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    statistics_detail = 'data/result/statistics_reactive/double_para'
    statistics_folder=os.path.join(parent_dir,statistics_detail)
    df_criterion = pd.read_excel(os.path.join(statistics_folder, 'double_criterion_100.xlsx'))
    df_sample= pd.read_excel(os.path.join(statistics_folder, 'functional_reactive_V01_20241231.xlsx'))
    results = calculate_reactive_percentage(df_criterion, df_sample)
    results_df=pd.DataFrame(results)
    atoms_number_excel = f'functional_reactive_atoms_error.xlsx'  # save functional group path
    statistics_atoms_number_path = os.path.join(statistics_folder, atoms_number_excel)
    results_df.to_excel(statistics_atoms_number_path, index=False)

