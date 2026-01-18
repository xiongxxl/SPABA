import os
import numpy as np
import pandas as pd

def drop_two_heads():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    folder_statistics_combine = os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244/combine')
    folder_statistics_heads = os.path.join(folder_statistics_combine, 'two_heads')

    atoms_error=['0','2','4']
    df_smiles_0=pd.DataFrame()
    df_smiles_2=pd.DataFrame()
    df_smiles_4=pd.DataFrame()
    merged_df=pd.DataFrame()
    for atoms in atoms_error:

        # atoms='0'
        excel_atoms_4_5 = f'4_5_{atoms}.xlsx'
        excel_atoms_7_7 = f'7_7_{atoms}.xlsx'
        excel_atoms_combine = f'combine_{atoms}.xlsx'
        excel_atoms_detail_4_5 = os.path.join(folder_statistics_heads, excel_atoms_4_5)
        excel_atoms_detail_7_7 = os.path.join(folder_statistics_heads, excel_atoms_7_7)
        excel_atoms_detail_combine = os.path.join(folder_statistics_heads, excel_atoms_combine)


        if os.path.exists(excel_atoms_detail_4_5) and os.path.exists(excel_atoms_detail_7_7):
            df1 = pd.read_excel(excel_atoms_detail_4_5)
            df2 = pd.read_excel(excel_atoms_detail_7_7)
            merged_df = pd.concat([df1, df2]).drop_duplicates()  # 合并并去重
            merged_df.to_excel(excel_atoms_detail_combine)

        elif os.path.exists(excel_atoms_detail_4_5):
            merged_df = pd.read_excel(excel_atoms_detail_4_5)
            merged_df.to_excel(excel_atoms_detail_combine)

        elif os.path.exists(excel_atoms_detail_7_7):
            merged_df = pd.read_excel(excel_atoms_detail_7_7)
            merged_df.to_excel(excel_atoms_detail_combine)

    return merged_df

if __name__ == "__main__":
    merged_df=drop_two_heads()