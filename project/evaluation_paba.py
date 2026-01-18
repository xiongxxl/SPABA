import pandas as  pd
from calculate_evaluation_metrics import compute_metrics_tensor
import os
import torch
import ast
import numpy as np

def calculate_reactive_percentage(df_criterion, df_sample):
    df_metrics_dict_single = pd.DataFrame()
    df_criterion_axis = df_criterion[['reactant', 'top1_top5_merge_deep']]

    for index_criterion, row_criterion in df_criterion_axis.iterrows():

        value_smiles_criterion = row_criterion['reactant']
        try:
            value_location_criterion_tensor =torch.tensor(eval(row_criterion['top1_top5_merge_deep']))
            filtered_df = df_sample[df_sample['reactant'] == value_smiles_criterion]
            predicted_atoms=filtered_df['reactive_atoms_deep']
            data_str = predicted_atoms.iloc[0]  # 获取第一个元素（字符串形式的列表）
            data_list = ast.literal_eval(data_str)  # 安全地将字符串转换为列表
            predicted_atoms_tensor = torch.tensor(data_list)
            metrics_dict =compute_metrics_tensor(value_smiles_criterion, value_location_criterion_tensor, predicted_atoms_tensor)
        except:
            pass
        df_metrics_dict = pd.DataFrame(metrics_dict)
        df_metrics_dict_single = pd.DataFrame(pd.concat([df_metrics_dict_single,df_metrics_dict], ignore_index=True))

    return df_metrics_dict_single

if __name__== "__main__":


### deal  uspto
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    input_folder_name = f'smiles/data/result/statistics_reactive/compare/uspto/rexgen'
    input_sample_file = os.path.join(parent_dir, input_folder_name)
    label_excel = os.path.join(input_sample_file, 'uspto_yang_transformer_7_7_50k_test_split_rexgen_merged_minus_deep.xlsx')
    df_criterion=pd.read_excel(label_excel)
    sample_excel= os.path.join(input_sample_file, 'uspto_yang_transformer_7_7_50k_test_split_rexgen_merged_minus_deep.xlsx')
    df_sample=pd.read_excel(sample_excel)
    df_metrics_dict_single=calculate_reactive_percentage(df_criterion, df_sample)
    metrics_excel=os.path.join(input_sample_file, 'uspto_yang_transformer_7_7_50k_test_split_rexgen_merged_minus_deep_evaluation.xlsx')
    df_metrics_dict_single.to_excel(metrics_excel)


# ### deal other method
#
#     current_dir = os.getcwd()
#     parent_dir = os.path.dirname(os.path.dirname(current_dir))
#     input_folder_name = f'data/result/statistics_supervision/uspto_yang/compare/uspto/'
#     input_sample_file = os.path.join(parent_dir, input_folder_name)
#     label_excel = os.path.join(input_sample_file, 'USPTO_50K_mark_1000_split_atoms_number_1098_function_deep.xlsx')
#     df_criterion=pd.read_excel(label_excel)
#     sample_excel= os.path.join(input_sample_file, 'USPTO_50K_mark_1000_split_atoms_number_1098_function_deep.xlsx')
#     df_sample=pd.read_excel(sample_excel)
#     df_metrics_dict_single=calculate_reactive_percentage(df_criterion, df_sample)
#     metrics_excel=os.path.join(input_sample_file, 'USPTO_50K_mark_1000_split_atoms_number_1098_function_deep_evaluation.xlsx')
#     df_metrics_dict_single.to_excel(metrics_excel)