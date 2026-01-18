import pandas as pd
import os
from tokenizer import tokenize_smiles


def auto_generation_label(df_atoms_single_smiles,df_atoms_single_label):

    positions=df_atoms_single_label
    filename_without_extension_re = tokenize_smiles(df_atoms_single_smiles)
    indexed_chars = [(i, char) for i, char in enumerate(filename_without_extension_re)]
    smiles_length=len(indexed_chars)
    filtered_chars = [(i, char) for i, char in indexed_chars if char.isalpha()]

    selected = [filtered_chars[i] for i in positions]
    indices = [item[0] for item in selected]

    result = [0] * smiles_length
    # indices = [0, 5, 13]
    for i in indices:
        result[i] = 1
    # print(result)
    return result

if __name__=="__main__":

    ##single sample
    df_criterion_smiles='CC(/C=C(C1=CC=CC=C1)\C2=CC=CC=C2)(C)/N=C/C3=CC=CC=C3'
    df_criterion_label=[7,9]
    label_list = auto_generation_label(df_criterion_smiles, df_criterion_label)
    print(label_list)


    # # all sample
    # head='combine'
    # current_dir = os.getcwd()
    # parent_dir = os.path.dirname(os.path.dirname(current_dir))
    # label_file_name = f'data/middle_attention/uspto_sample/{head}/label'
    # label_excel = os.path.join(parent_dir, label_file_name)
    # df_criterion=pd.read_excel(os.path.join(label_excel, 'USPTO_50K_mark_1000_split_atoms_number_1244_gold.xlsx'))
    #
    # for index_sample,row_sample in df_criterion.iterrows():
    #     df_criterion_smiles = row_sample['smiles']
    #     df_criterion_label = eval(row_sample['gold_criterion'])
    #     print(df_criterion_smiles)
    #     label_list =auto_generation_label(df_criterion_smiles, df_criterion_label)
    #     df_criterion.at[index_sample,'gold_criterion_deep']=str(label_list)
    #
    # df_criterion.to_excel(os.path.join(label_excel, 'USPTO_50K_mark_1000_split_atoms_number_1244_gold_label.xlsx'))


# ## deal with excel
#
#     current_dir = os.getcwd()
#     parent_dir = os.path.dirname(os.path.dirname(current_dir))
#     file_name = 'data/result/statistics_supervision/uspto_deep/overview/statistic'
#     input_files = os.path.join(parent_dir, file_name)
#     excel_name = f'USPTO_50K_mark_1000_split_atoms_number_1098_function.xlsx'
#     # excel_path = os.path.join(input_files, excel_name)
#     excel_path='SMLES_ruizhen.xlsx'
#     df_criterion=pd.read_excel(excel_path)
#
#     for index_sample,row_sample in df_criterion.iterrows():
#         df_criterion_smiles = row_sample['reactant']
#         df_criterion_label = eval(row_sample['reactive_atoms'])
#         print(df_criterion_smiles)
#         label_list =auto_generation_label(df_criterion_smiles, df_criterion_label)
#         df_criterion.at[index_sample,'reactive_atoms_deep']=str(label_list)
#
#     excel_name_deep=f'USPTO_50K_mark_1000_split_atoms_number_1098_function_deep.xlsx'
#     # excel_output_path=os.path.join(input_files, excel_name_deep)
#     excel_output_path='SMLES_ruizhen_deep.xlsx'
#     df_criterion.to_excel(excel_output_path)

