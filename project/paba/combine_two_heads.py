## save 4_5 ,7_7 attention by reactive result

from copy_flies_by_path import copy_and_rename_files
import numpy as np
import pandas as pd
import os

def combine_two_heads():
    ##create new files
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    folder_statistics_combine = os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244/combine')
    folder_statistics_heads=os.path.join(folder_statistics_combine,'two_heads')
    data_files_img = 'data/result/img_supervision/uspto_1244/combine'
    folder_name_attn = os.path.join(parent_dir, data_files_img)
    folder_name_attn_npy=os.path.join(folder_name_attn,'64_heads')
    folder_name_attn_combine=os.path.join(folder_name_attn, 'combine')

    #create files
    # create two files
    atoms_error=['0','2','4']
    head_list=['4_5','7_7']
    for atoms in atoms_error:
        for head in head_list:
            atoms_error_file = f'{atoms}_atmos_error_{head}'
            atoms_error_folder_0 = os.path.join(folder_name_attn_combine,atoms_error_file)
            if not os.path.exists(atoms_error_folder_0):
                os.makedirs(atoms_error_folder_0)

    head_list=['4_5','7_7']
    df_atoms=pd.DataFrame()

    for head in head_list:
            # head='7_7'
        atoms_excel=f'df_atoms_error_{head}.xlsx'
        df_atoms_error = pd.read_excel(os.path.join(folder_statistics_combine, atoms_excel))

        atoms_error=['0','2','4']
        for atoms in atoms_error:
            atoms=eval(atoms)
            # atoms=4
            if atoms ==0:
                index = f'flag_{atoms}'
                df_atoms=df_atoms_error[df_atoms_error[index]==1]
            if atoms ==2:
                index_2= f'flag_{atoms}'
                atoms_1= atoms-1
                index_1= f'flag_{atoms_1}'
                atoms_0= atoms-2
                index_0= f'flag_{atoms_0}'
                df_atoms = df_atoms_error[(df_atoms_error[index_2] == 1)|(df_atoms_error[index_1]==1)|(df_atoms_error[index_0]==1)]
            if atoms == 4:
                index_4= f'flag_{atoms}'
                atoms_3=atoms-1
                index_3= f'flag_{atoms_3}'
                atoms_2=atoms-2
                index_2=f'flag_{atoms_2}'
                atoms_1=atoms-3
                index_1= f'flag_{atoms_1}'
                atoms_0=atoms-4
                index_0=f'flag_{atoms_0}'
                df_atoms = df_atoms_error[(df_atoms_error[index_4] == 1)|(df_atoms_error[index_3] == 1)|(df_atoms_error[index_2] == 1)
                                           |(df_atoms_error[index_1] == 1)|(df_atoms_error[index_0] == 1)]


            if df_atoms.empty:
                pass
                # df_atmos_smiles.to_excel('df_atmos_smiles_4_5_4.xlsx')
            else:

                df_atoms_smiles = df_atoms['smiles']
                excel_atoms=f'{head}_{atoms}.xlsx'
                excel_atoms_detail=os.path.join(folder_statistics_heads,excel_atoms)
                df_atoms_smiles.to_excel(excel_atoms_detail)

                # for smiles in df_atoms_smiles:
                #     smiles_file_name=f'{smiles}_0.99'
                #     src_folder_npy=os.path.join(folder_name_attn_npy,smiles_file_name)
                #     attn_file_name = f'{atoms}_atmos_error_{head}'
                #     dest_folder=os.path.join(folder_name_attn_combine,attn_file_name)
                #
                #     file_names_npy=f'{head}.npy'
                #     new_names_npy=f'{smiles}_{head}.npy'
                #     copy_and_rename_files(src_folder_npy, dest_folder, file_names_npy, new_names_npy)
                #
                #     file_names_png=f'{head}.png'
                #     new_names_png=f'{smiles}_{head}.png'
                #     copy_and_rename_files(src_folder_npy, dest_folder, file_names_png, new_names_png)
                #
                #     ## copy img result
                #     file_names_img=f'{smiles}_{head}_0.99.jpeg'
                #     src_folder_img=folder_name_attn_combine
                #     copy_and_rename_files(src_folder_img, dest_folder, file_names_img, file_names_img)
    return  df_atoms_smiles

if __name__ == "__main__":
    src_folder_img = combine_two_heads()


