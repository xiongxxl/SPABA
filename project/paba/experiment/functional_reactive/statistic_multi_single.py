
import pandas as pd
import os
import numpy as np
from collections import Counter
import re

def extract_numbers(text):
    return re.findall(r'\d+_\d+', text)

def combine_multi_single_column(df_atoms_error):

    df = pd.DataFrame(df_atoms_error)
    # 计算 flag_0 到 flag_5 列为 1 的个数
    flag_columns = [f'flag_{i}' for i in range(6)]


    flag_count = (df[flag_columns] == 1).sum(axis=0)
    # 合并 head_axis_0 到 head_axis_3 列的数据
    head_columns = [f'head_axis_{i}' for i in range(5)]
    head_combined = df[head_columns].apply(lambda x: ','.join(x.astype(str)), axis=1)

    df_single_row= pd.DataFrame([flag_count.tolist() + [','.join(head_combined)]],
                          columns=['flag_0', 'flag_1', 'flag_2', 'flag_3', 'flag_4', 'flag_5', 'head_combined'])

    df_single_row['head_combined']=df_single_row['head_combined'].str.replace("[","").str.replace("]","").str.replace(",","")
    df_single_row['head_combined_re'] = df_single_row['head_combined'].apply(extract_numbers)
    all_numbers = [item for sublist in df_single_row['head_combined_re'] for item in sublist]
    counter = Counter(all_numbers)  # Get the top 5 most common numbers
    ##top3
    top_3 = counter.most_common(3)
    top_3_str = ', '.join([f"{num}: {count}" for num, count in top_3])
    top_3_no_number=', '.join([f"{num}" for num, count in top_3])
    df_single_row.at[0,'head_combined_rank']=top_3_str
    df_single_row.at[0, 'head_combined_no_number']=top_3_no_number
    # ##top5
    # top_5 = counter.most_common(5)
    # top_5_str = ', '.join([f"{num}: {count}" for num, count in top_5])
    # df_single_row.at[0,'head_combined_rank']=top_5_str

    return df_single_row

if __name__=="__main__":

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    statistics_detail = 'data/result/statistics_reactive/double_para'
    statistics_folder=os.path.join(parent_dir,statistics_detail)
    atoms_number_excel = f'functional_reactive_atoms_error.xlsx'  # save functional group path
    statistics_atoms_number_path = os.path.join(statistics_folder, atoms_number_excel)
    df_atoms_error=pd.read_excel(statistics_atoms_number_path)
    df_single_row= combine_multi_single_column(df_atoms_error)

    single_row_excel = f'df_single_row.xlsx'  # save functional group path
    single_row_number_path = os.path.join(statistics_folder, single_row_excel)
    df_single_row.to_excel(single_row_number_path)


    # df_combine_attention_split=df_combine_attention['combine_attention'].str.split(' ', expand=True)
    # df_combine_attention_split=df_combine_attention_split.T
    # df_combine_attention_split.columns=['head_axis']
    # df_combine_attention_split=df_combine_attention_split['head_axis'].str.replace("[","").str.replace("]","").str.replace(",", "")
    # df_combine_attention_split.replace("",np.nan,inplace=True)
    # df_combine_attention_split.dropna()
    # top_rank_head = df_combine_attention_split.value_counts().head(5)
    # top_rank_head_dict=top_rank_head.to_dict()
    # top_rank_head_str= ','.join(f"{key}:{value}" for key, value in top_rank_head_dict.items())
    # df_rank_head.at[0,'rank_head']=top_rank_head_str

    # ##find top-ranking attention
    # df_single_row = pd.DataFrame()
    # df_combine_attention=pd.DataFrame()
    # filtered_reactive_flag=pd.DataFrame()
    # filtered_head_flag=pd.DataFrame()
    # df_rank_head=pd.DataFrame()
    #
    # for i in range(6):
    #     flag_index=f'flag_{i}'
    #     reactive_index=f'reactive_atoms_{i}'
    #     head_index=f'head_axis_{i}'
    #
    #     df_single_row.loc[0, flag_index]=(df_atoms_error[flag_index]==1).sum()
    #
    #     filtered_reactive_flag = df_atoms_error[df_atoms_error[flag_index]==1]
    #     df_single_row.loc[0, reactive_index] =filtered_reactive_flag[reactive_index].str.cat(sep=' ')
    #     filtered_head_flag = df_atoms_error[df_atoms_error[flag_index] == 1]
    #     df_single_row.loc[0, head_index]= filtered_head_flag[head_index].str.cat(sep=' ') #conbine list
    #     df_combine_attention['combine_attention']=df_single_row['head_axis_0']+' '+df_single_row['head_axis_1']+' '+df_single_row['head_axis_2']
    #     df_rank_head=df_single_row[['flag_0','flag_1','flag_2','flag_3','flag_4','flag_5']]
    #     df_combine_attention_split=df_combine_attention['combine_attention'].str.split(' ', expand=True)
    #     df_combine_attention_split=df_combine_attention_split.T
    #     df_combine_attention_split.columns=['head_axis']
    #     df_combine_attention_split=df_combine_attention_split['head_axis'].str.replace("[","").str.replace("]","").str.replace(",", "")
    #     df_combine_attention_split.replace("",np.nan,inplace=True)
    #     df_combine_attention_split.dropna()
    #     top_rank_head = df_combine_attention_split.value_counts().head(5)
    #     top_rank_head_dict=top_rank_head.to_dict()
    #     top_rank_head_str= ','.join(f"{key}:{value}" for key, value in top_rank_head_dict.items())
    #     df_rank_head.at[0,'rank_head']=top_rank_head_str







