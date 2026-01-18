import os
import numpy as np
import pandas as pd
from restrain_functional_reactive import restrain_functional_axis
from comparison_reactive import calculate_reactive_percentage
from statistic_multi_single import combine_multi_single_column


current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
filename_combine= os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244_bio/find_alpha')
df_frag_smiles_multi_4_5=pd.read_excel(os.path.join(filename_combine, 'df_frag_smiles_multi_4_5.xlsx'))
df_frag_smiles_multi_7_7=pd.read_excel(os.path.join(filename_combine, 'df_frag_smiles_multi_7_7.xlsx'))

df_frag_smiles_multi_combine=pd.DataFrame()

reactive_atoms = []
df_frag_smiles_multi_7_7.set_index('smiles',inplace=True)

for index_sample, row_sample in df_frag_smiles_multi_4_5.iterrows():
    row_sample_smiles=row_sample['smiles']
    reactive_atoms_4_5 =eval(row_sample['reactive_atoms'])
    reactive_atoms_7_7=eval(df_frag_smiles_multi_7_7.loc[row_sample_smiles,'reactive_atoms'])

    smiles=row_sample_smiles
    results, reactive_atoms_restrain_result, less_dot_criterion, greater_equal_dot_criterion = restrain_functional_axis(
        smiles, reactive_atoms)

    if  len(reactive_atoms_4_5)>0 and len(reactive_atoms_7_7)==0:
        df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] =str(reactive_atoms_4_5)

    elif len(reactive_atoms_4_5)==0 and len(reactive_atoms_7_7)>0 :
        df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] =str(reactive_atoms_7_7)
        df_frag_smiles_multi_4_5.at[index_sample, 'head_axis'] = df_frag_smiles_multi_7_7.loc[row_sample_smiles,'head_axis']
    elif  len(reactive_atoms_4_5)==0  and len(reactive_atoms_7_7)==0 :
        df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] =str(reactive_atoms_4_5)

    else:
        if set(reactive_atoms_4_5)&set(less_dot_criterion) and set(reactive_atoms_4_5)&set(greater_equal_dot_criterion):
             df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] =str(reactive_atoms_4_5)

        elif set(reactive_atoms_7_7)&set(less_dot_criterion) and set(reactive_atoms_7_7)&set(greater_equal_dot_criterion):
            df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] = str(reactive_atoms_7_7)
            df_frag_smiles_multi_4_5.at[index_sample, 'head_axis'] = df_frag_smiles_multi_7_7.loc[row_sample_smiles, 'head_axis']

        else:
            df_frag_smiles_multi_4_5.at[index_sample, 'combine_atoms'] =str(reactive_atoms_4_5)


current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
filename_predict = os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244_bio/predict')
excel_combine_path=os.path.join(filename_predict, 'df_frag_smiles_multi_predict.xlsx')
df_frag_smiles_multi_4_5.to_excel(excel_combine_path)
df_frag_smiles_multi_combine['smiles']=df_frag_smiles_multi_4_5['smiles']
df_frag_smiles_multi_combine['reactive_atoms']=df_frag_smiles_multi_4_5['combine_atoms']
df_frag_smiles_multi_combine['head_axis']=df_frag_smiles_multi_4_5['head_axis']


df_criterion = pd.read_excel(os.path.join(filename_combine, 'USPTO_50K_mark_1000_split_atoms_number.xlsx'))
df_atoms_error = calculate_reactive_percentage(df_criterion, df_frag_smiles_multi_combine)
df_single_row = combine_multi_single_column(df_atoms_error)
df_single_row.to_excel(os.path.join(filename_predict, 'df_single_row_predict.xlsx'))









