import pandas as pd
import os
import re


def extract_numbers(atoms):
    # 使用正则表达式找到所有的数字
    numbers = re.findall(r'\d+', atoms)
    return numbers

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_parent_dir=os.path.dirname(parent_dir)
input_files_smiles='data/input_smiles/reaction'
output_files_critertion='data/result/statistics_functional_property/double_0911mark/'

foldername_input=os.path.join(parent_parent_dir, input_files_smiles)
df = pd.read_excel((os.path.join(foldername_input, '0911mark.xlsx')), header=0,index_col=0)

foldername_output=os.path.join(parent_parent_dir, output_files_critertion)
df['criterion'] = df['Marked Atoms'].apply(lambda x: extract_numbers(x))

df['criterion'] = df['criterion'].apply(pd.to_numeric,errors='coerce')
output_path = os.path.join(foldername_output, '0911mark_number_atoms_criterion.xlsx')
df.to_excel(output_path, index=False)




