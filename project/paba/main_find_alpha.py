import numpy as np
import pandas as pd
from tokenizer import tokenize_smiles
import sys
from binary_highatom_reactive_single import highlight_chemical_atoms_single
from annotate_num_chemical import annotate_atoms
from syn_1_img import syn_1_jpg
from syn_1_img import syn_1_jpeg
from syn_1_img import syn_1_png
from attn_binary import attn_64_img
import os
import time
from comparison_reactive import calculate_reactive_percentage
from statistic_multi_single import combine_multi_single_column

head='7_7'
saving_img_flag = 0## flag=0 is not saving picture,flag=1 is saving picture
## calculate single alpha
single_ratio_flag=1  ##when flag is 0,ratio run 0.97 to 0.999.when flag is 1 ,then ratio run single value.
ratio_adaptive_fix=0.986

df_ratio_multi=pd.DataFrame()
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
input_folder_part = 'data/result/statistics_supervision/name_1000'
input_folder_all=os.path.join(parent_dir, input_folder_part)
ratio_excel = f'df_ratio_multi_0.900_0.999_{head}.xlsx'  # save functional group path
statistics_path=os.path.join(input_folder_all,'find_alpha')
statistics_ratio_path = os.path.join(statistics_path, ratio_excel)
folder_ratio_statistics = os.path.join(parent_dir, statistics_ratio_path)


for ratio_adaptive_change in np.arange(0.960, 0.999, 0.001):
    if single_ratio_flag:
        ratio_adaptive = ratio_adaptive_fix
    else:
        ratio_adaptive = ratio_adaptive_change
    data_files_npy='data/middle_attention/npy_supervision/name_1000/del_1000_v04/64_attention'
    folder_name_npy=os.path.join(parent_dir, data_files_npy)
    data_files_img='data/result/img_supervision/name_1000/find_alpha'
    folder_name_img=os.path.join(parent_dir, data_files_img)

    #save fragments
    fragment_excel=f'frag_smiles_main_{ratio_adaptive}.xlsx'  #save functional group path
    statistics_fragment_path= os.path.join(statistics_path, fragment_excel)
    folder_fragment_statistics=os.path.join(parent_dir, statistics_fragment_path)

    atoms_number_excel=f'df_ratio_multi_{ratio_adaptive}.xlsx'  #save functional group path
    statistics_atoms_number_path= os.path.join(statistics_path, atoms_number_excel)
    folder_atoms_number_statistics=os.path.join(parent_dir, statistics_atoms_number_path)


    df_frag_smiles_multi=pd.DataFrame()
    df_atoms_number_multi =pd.DataFrame()
    j=0
    for filename in os.listdir(folder_name_npy):
        if filename.endswith('.npy'):
            filepath = os.path.join(folder_name_npy, filename)
            attn = np.load(filepath)
            filename_without_extension = os.path.splitext(filename)[0]
            if 'H' in filename_without_extension:
                j=j+1
                pass
            else:
                ratio=ratio_adaptive
                # data_files_npy_simple='data/middle_attention/npy_supervision/multi_1000_double_20_atoms/CC1=NNC=C1.O=C(C)O[Pb](OC(C)=O)(OC(C)=O)C2=CC=CC=C2.npy'
                # attn=np.load(os.path.join(parent_dir, data_files_npy_simple))
                # filename_without_extension='CC1=NNC=C1.O=C(C)O[Pb](OC(C)=O)(OC(C)=O)C2=CC=CC=C2'
                smiles_address=filename_without_extension  #this filename for saving address
                filename_without_extension = filename_without_extension.replace("x", "/")
                filename_without_extension_re= tokenize_smiles(filename_without_extension)  #re syn like cl to one element
                j=j+1
                print(f'Filename: {filename_without_extension}')
                print(j)

                start_time1=time.time()
                img_64_filename,file_path_decrease_one_npy,file_path_decrease_two_npy=attn_64_img(attn, folder_name_img,
                                                    filename_without_extension,ratio,saving_img_flag)#produce 64 attn image
                if saving_img_flag:
                    syn_name_attn=f'{smiles_address}_{ratio}.jpg'
                    output_image_attn=os.path.join(folder_name_img, syn_name_attn)
                    attn_syn_img=syn_1_jpg(img_64_filename, output_image_attn) #syn 64 attention to 1 image
                    print(img_64_filename)

                end_time1=time.time()
                # print(f"代码1运行时间：{end_time1-start_time1}秒")

                if saving_img_flag:
                    #syn 64 binary attention to  1 image
                    suffix_b = '_binarize'
                    syn_name_binarize=f'{smiles_address}{suffix_b}_{ratio}.jpg'
                    output_image_binarize=os.path.join(folder_name_img, syn_name_binarize)
                    attn_syn_binarize=syn_1_png(img_64_filename, output_image_binarize)

                ##syn 64 highatom images
                start_time2 = time.time()
                df_frag_smile_single = highlight_chemical_atoms_single(img_64_filename,file_path_decrease_one_npy,
                        file_path_decrease_two_npy,filename_without_extension,saving_img_flag,ratio,smiles_address,head)

                df_frag_smiles_multi=pd.DataFrame(pd.concat([df_frag_smiles_multi,df_frag_smile_single],ignore_index=True))
                end_time2 = time.time()
                # print(f"代码2运行时间：{end_time2 - start_time2}秒")

                if saving_img_flag:
                    suffix_c = '_lightatom'
                    syn_name_atom=f'{smiles_address}{suffix_c}_{ratio}.jpg'
                    output_image_atom=os.path.join(folder_name_img, syn_name_atom)
                    attn_syn_atom=syn_1_jpeg(img_64_filename, output_image_atom)


    if single_ratio_flag:
        frag_multi_excel=f'df_frag_smiles_multi_{head}.xlsx'
        df_frag_smiles_multi.to_excel(os.path.join(statistics_path, frag_multi_excel))

    df_criterion = pd.read_excel(os.path.join(input_folder_all, 'name_with_1000 _del_complex_v04_20250503.xlsx'))
    df_atoms_error = calculate_reactive_percentage(df_criterion, df_frag_smiles_multi)

    if single_ratio_flag:
        frag_multi_excel=f'df_atoms_error_{head}.xlsx'
        df_atoms_error.to_excel(os.path.join(statistics_path, frag_multi_excel))

    df_single_row=combine_multi_single_column(df_atoms_error)
    df_atoms_number_multi = pd.DataFrame(pd.concat([df_atoms_number_multi, df_single_row], ignore_index=True))
    df_ratio = pd.DataFrame()
    df_ratio.at[0, 'ratio'] = ratio
    df_ratio.at[0, '0_number'] = df_atoms_number_multi.iloc[0,0]
    df_ratio.at[0, '1_number'] = df_atoms_number_multi.iloc[0,1]
    df_ratio.at[0, '2_number'] = df_atoms_number_multi.iloc[0,2]
    df_ratio.at[0, '3_number'] = df_atoms_number_multi.iloc[0,3]
    df_ratio.at[0, '4_number'] = df_atoms_number_multi.iloc[0,4]
    df_ratio.at[0, '5_number'] = df_atoms_number_multi.iloc[0,5]
    df_ratio.at[0,'head_rank']= df_atoms_number_multi.iloc[0,8]
    df_ratio.at[0, 'head_rank_no_number'] = df_atoms_number_multi.iloc[0, 9]
    df_ratio_multi = pd.DataFrame(pd.concat([df_ratio_multi, df_ratio], ignore_index=True))
    if single_ratio_flag:
        ratio_multi_excel=f'df_ratio_multi_{head}.xlsx'
        df_ratio_multi.to_excel(os.path.join(statistics_path, ratio_multi_excel))
        sys.exit()
df_ratio_multi.to_excel(folder_ratio_statistics)







