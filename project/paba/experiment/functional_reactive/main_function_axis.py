## find all smiles functional group axis
##by xxl 20250103
import pandas as pd
import os
from find_functional_reactive import  find_functional_axis
from comparison_reactive_atoms import calculate_reactive_percentage
from statistic_multi_single import combine_multi_single_column

df_function_reactive=pd.DataFrame()
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
statistics_path='data/result/statistics_supervision/multi_1000_double_20_atoms'
folder_statistics_path=os.path.join(parent_dir, statistics_path)
criterion_excel=f'multi_criterion_1000_double_20_atoms_v01.xlsx'  #save functional group path
criterion_path= os.path.join(folder_statistics_path, criterion_excel)
df_criterion=pd.read_excel(criterion_path)
df_criterion['reactive_atoms']=df_criterion['smiles'].apply(find_functional_axis)
df_criterion['head_axis']=pd.DataFrame
function_excel=f'df_function_reactive.xlsx'  #save functional group path
folder_statistics_predict=os.path.join(folder_statistics_path,'predict')
function_path= os.path.join(folder_statistics_predict, function_excel)
df_criterion.to_excel(function_path)

df_sample=pd.read_excel(function_path)
results = calculate_reactive_percentage(df_criterion, df_sample)
results_df = pd.DataFrame(results)
atoms_number_excel = f'functional_reactive_atoms_error.xlsx'  # save functional group path
statistics_atoms_number_path = os.path.join(folder_statistics_predict, atoms_number_excel)
results_df.to_excel(statistics_atoms_number_path, index=False)


df_atoms_error = pd.read_excel(statistics_atoms_number_path)
df_single_row = combine_multi_single_column(df_atoms_error)

single_row_excel = f'df_single_row.xlsx'  # save functional group path
single_row_number_path = os.path.join(folder_statistics_predict, single_row_excel)
df_single_row.to_excel(single_row_number_path)
