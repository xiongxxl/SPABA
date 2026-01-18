import numpy as np
import os
from tokenizer import tokenize_smiles
from highlight_num_atoms import highlight_num_atoms
from remove_letter_number import find_and_remove_non_letters
from remove_letter_number import find_non_alphanumeric_positions_and_remove
from exact_smile_fragment import extract_specified_atoms
from find_functional_group import identify_functional_groups
import pandas as pd
from restrain_functional_reactive import restrain_functional_axis_single
from PIL import Image
import io

from restrain_functional_reactive import restrain_functional_axis
from annotate_num_chemical import annotate_atoms
from decrease_ratio_attn import decrease_ratio_highlight_atoms
def remove_rows_and_columns(matrix, rows_to_remove, cols_to_remove):
    # 转换输入列表为NumPy数组
    matrix_np = np.array(matrix)
    # 删除指定的行
    matrix_np = np.delete(matrix_np, rows_to_remove, axis=0)
    # 删除指定的列
    matrix_np = np.delete(matrix_np, cols_to_remove, axis=1)
    return matrix_np

#folder_path = './result/attention/first/img/CCCCCCCCCCCCCCO'
# 遍历根目录下的所有文件夹
#df_frag_smiles = pd.DataFrame(columns=["simles","frag","functional_group","location","attention_axis"])


def highlight_chemical_atoms(folder_path,filename_without_extension,saving_img_flag,ratio,smiles_address):
    if 'H' in filename_without_extension:
        pass
    else:
        df_frag_smile_single=pd.DataFrame()
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(folder_path, file_name)
                #file_path='/mnt/work/code/tian/smiles/data/result/img_reactive/old/double_0911mark/CCC(=O)CC(=O)C(F)(F)F.Nc1ccccc1Br_0.96/5_4.npy'

                attn_binary = np.load(file_path)
                #np.fill_diagonal(attn_binary,0) #set diagonal element to 0
                attn_binary_del=attn_binary
                # del non-character
                filename_without_extension_re = tokenize_smiles(filename_without_extension)
                ## delete alpha
                #positions, cleaned_string = find_non_alphanumeric_positions_and_remove(filename_without_extension_re)
                positions, cleaned_string = find_and_remove_non_letters(filename_without_extension_re)
                # np.save(f'{filename_without_extension}.npy',attn_binary_del)
                # print('cleaned_string is:',cleaned_string)
                # print('positions is:', positions)
                # print('attn_binary :', np.shape(attn_binary))
                # print(attn_binary_del)

                # del Delete the corresponding rows and columns
                attn_binary_letter = remove_rows_and_columns(attn_binary_del, positions, positions)
                #print('attn_binary_del:', np.shape(attn_binary_letter))

                # find no_zero_element
                non_zero_positions = np.nonzero(attn_binary_letter)
                non_zero_col=np.unique(non_zero_positions[0]).tolist()
                ##concatatenate nadarry  and del copy
                # non_zero_positions_combined = np.concatenate(non_zero_positions[0], non_zero_positions[1])
                # non_zero_positions_combined_unique = np.unique(non_zero_positions_combined)

                #non_zero_row=np.unique(non_zero_positions[1]).tolist()
                # print("不为零元素的行索引：", non_zero_col)
                # print("不为零元素的列索引：", non_zero_row)

                ## find functonal group
                smiles = filename_without_extension
                atom_indices = non_zero_col
                # results,reactive_atoms_restrain_last,less_dot_criterion, greater_equal_dot= restrain_functional_axis(smiles, atom_indices)
                results,reactive_atoms_restrain_last=restrain_functional_axis_single(smiles, atom_indices)
                atom_indices_restrain=list(set(reactive_atoms_restrain_last))
                keep_single_elements, del_single_elements, frag_smiles = extract_specified_atoms(smiles, atom_indices_restrain)
                functional_groups_state, functional_groups_detail= identify_functional_groups(frag_smiles, smiles)

                file_name_nosuffix = os.path.splitext(file_name)[0]
                frag_smiles_dict= {
                                      'smiles' :[filename_without_extension],
                                 'frag_smiles' :[frag_smiles],
                            'functional_group' :[functional_groups_detail],
                              'reactive_atoms' :[str(atom_indices_restrain)],
                                   'head_axis' :[file_name_nosuffix],
                                   }

                df_frag_smile_attention=pd.DataFrame(frag_smiles_dict)
                df_frag_smile_single=pd.DataFrame(pd.concat([df_frag_smile_single,df_frag_smile_attention],ignore_index=True))

                if saving_img_flag:
                    #atom_indices = non_zero_row
                    #print(atom_indices)
                    img_data = highlight_num_atoms(smiles, atom_indices_restrain)
                    file_name_no_suffix=os.path.splitext(file_name)[0]
                    img_filename=f'{file_name_no_suffix}.jpeg'
                    img_path=os.path.join(folder_path,img_filename)
                    #img = Image.open(io.BytesIO(img_data))
                    #img.show()
                    with open(img_path, "wb") as f:
                        f.write(img_data)

                    ##save (4,5)attention
                    #
                    # if file_name_no_suffix=='4_5':
                    #     predict_file ='predict'
                    #     parent_folder_path=os.path.dirname(folder_path)
                    #     predict_folder = os.path.join(parent_folder_path, predict_file)
                    #     if not os.path.exists(predict_folder):
                    #         os.makedirs(predict_folder)
                    #
                    #     annotated_smiles,img,numbers_before_dot,numbers_after_dot=annotate_atoms(smiles)
                    #     if set(atom_indices_restrain).intersection(numbers_before_dot) and set(atom_indices_restrain).intersection(numbers_after_dot): #atom_indices both element
                    #         atom_indices_adaptive=atom_indices_restrain
                    #
                    #     else:
                    #         attn_axis=f'4_5.npy'
                    #         folder_path_decrease_one=os.path.join(file_path_decrease_one_npy,attn_axis)
                    #         reactive_atoms_restrain_one=decrease_ratio_highlight_atoms(folder_path_decrease_one, filename_without_extension)
                    #         ratio=ratio-0.01
                    #
                    #         if set(reactive_atoms_restrain_one).intersection(numbers_before_dot) and set(reactive_atoms_restrain_one).intersection(numbers_after_dot):
                    #             atom_indices_adaptive = reactive_atoms_restrain_one
                    #         else:
                    #             attn_axis = f'4_5.npy'
                    #             folder_path_decrease_two = os.path.join(file_path_decrease_two_npy, attn_axis)
                    #             reactive_atoms_restrain_two = decrease_ratio_highlight_atoms(folder_path_decrease_two,filename_without_extension)
                    #             atom_indices_adaptive = reactive_atoms_restrain_two
                    #             ratio = ratio -0.01
                    #
                    #             if set(reactive_atoms_restrain_one).intersection(numbers_before_dot) and set(reactive_atoms_restrain_one).intersection(numbers_after_dot):
                    #                 atom_indices_adaptive = reactive_atoms_restrain_one
                    #             else:
                    #                 attn_axis = f'4_5.npy'
                    #                 folder_path_decrease_two = os.path.join(file_path_decrease_two_npy, attn_axis)
                    #                 reactive_atoms_restrain_two = decrease_ratio_highlight_atoms(
                    #                     folder_path_decrease_two, filename_without_extension)
                    #                 atom_indices_adaptive = reactive_atoms_restrain_two
                    #                 ratio = ratio - 0.01
                        # #atom_indices_adaptive=atom_indices
                        # img_data = highlight_num_atoms(smiles, atom_indices_adaptive)
                        # file_name_no_suffix = os.path.splitext(file_name)[0]
                        # img_filename = f'{smiles_address}_{file_name_no_suffix}_{ratio}.jpeg'
                        # img_path = os.path.join(predict_folder, img_filename)
                        # with open(img_path, "wb") as f:
                        #     f.write(img_data)

        return df_frag_smile_single

if __name__ == "__main__":
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    data_files_npy_simple ='data/result/img_reactive/old/double_0911mark/CCC(=O)CC(=O)C(F)(F)F.Nc1ccccc1Br_0.96'
    folder_path =os.path.join(parent_dir, data_files_npy_simple)
    filename_without_extension='CCC(=O)CC(=O)C(F)(F)F.Nc1ccccc1Br'
    saving_img_flag=1
    file_path_decrease_one_npy=''
    file_path_decrease_two_npy=''
    ratio=0.98
    saving_img_flag=1
    smiles_address='CCC(=O)CC(=O)C(F)(F)F.Nc1ccccc1Br'
    #statistics_fragment_path= 'data/result/statistics_reactive/double_1028mark/frag_smiles_main_0.98.csv'
    #df_frag_smile_single=highlight_chemical_atoms(folder_path,filename_without_extension,saving_img_flag)
    df_frag_smile_single=highlight_chemical_atoms(folder_path, file_path_decrease_one_npy, file_path_decrease_two_npy,
                             filename_without_extension, saving_img_flag, ratio,smiles_address)
    print(df_frag_smile_single)

