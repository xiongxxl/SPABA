import sys
import numpy as np
import pandas as pd
from tokenizer import tokenize_smiles
from binary_highatom_reactive import highlight_chemical_atoms
from syn_1_img import syn_1_jpg
from syn_1_img import syn_1_jpeg
from syn_1_img import syn_1_png
from attn_binary import attn_64_img
import os
import time
from comparison_reactive import calculate_reactive_percentage
from statistic_multi_single import combine_multi_single_column

current_dir = os.getcwd()
parent_dir =os.path.dirname(os.path.dirname(current_dir))
statistics_path = 'data/result/statistics_supervision/uspto_1244/find_heads'
ratio_excel = f'df_ratio_multi_0.90_0.99.xlsx'  # save functional group path
statistics_ratio_path = os.path.join(statistics_path, ratio_excel)
folder_ratio_statistics = os.path.join(parent_dir, statistics_ratio_path)
df_ratio_multi=pd.DataFrame()

single_ratio_flag=0  ##when flag is 0,ratio run 0.97 to 0.999.when flag is 1 ,then ratio run single value.
ratio_adaptive_fix=0.99
head_axis='all'
saving_deep_sample = 0
saving_img_flag = 0



for ratio_adaptive_change in np.arange(0.90, 0.999, 0.001):
    if single_ratio_flag:
        ratio = ratio_adaptive_fix
    else:
        ratio = ratio_adaptive_change
    data_files_npy='data/middle_attention/npy_supervision/uspto_50k/uspto_1244/64_attention'
    folder_name_npy=os.path.join(parent_dir, data_files_npy)
    data_files_img='data/result/img_supervision/uspto_1244/find_heads'
    folder_name_img=os.path.join(parent_dir, data_files_img)

    # save fragments
    statistics_path='data/result/statistics_supervision/uspto_1244'
    fragment_excel=f'frag_smiles_main_{ratio}.xlsx'  #save functional group path
    statistics_fragment_path= os.path.join(statistics_path, fragment_excel)
    folder_fragment_statistics=os.path.join(parent_dir, statistics_fragment_path)

    statistics_path='data/result/statistics_supervision/uspto_1244'
    percentage_excel=f'frag_atoms_error_main_{ratio}.xlsx'  #save functional group path
    statistics_percentage_path= os.path.join(statistics_path, percentage_excel)
    folder_percentage_statistics=os.path.join(parent_dir, statistics_percentage_path)

    j = 0
    ## 0 is saving picture,1 isnot saving picture.
    df_frag_smiles_multi=pd.DataFrame()
    df_atoms_number_multi =pd.DataFrame()

    for filename in os.listdir(folder_name_npy):
        if filename.endswith('.npy'):
            filepath = os.path.join(folder_name_npy, filename)
            attn = np.load(filepath)

            filename_without_extension = os.path.splitext(filename)[0]
            if 'H' in filename_without_extension:
                j=j+1
                pass
            else:
                # data_files_npy_simple='data/middle_attention/npy_supervision/last_attention/multi_1000_test/BrC1(Br)CC1(C2=CC=CC=C2)C.npy'
                # attn=np.load(os.path.join(parent_dir, data_files_npy_simple))
                # filename_without_extension='BrC1(Br)CC1(C2=CC=CC=C2)C'
                smiles_address=filename_without_extension
                filename_without_extension = filename_without_extension.replace("x", "/")
                filename_without_extension_re= tokenize_smiles(filename_without_extension)  #re syn like cl to one element
                filename_without_extension_forsaving=filename_without_extension.replace("/", "x") #this filename for saving address
                j=j+1
                print(f'Filename: {filename_without_extension}')
                print(j)
                start_time1=time.time()
                img_64_filename=attn_64_img(attn, folder_name_img,filename_without_extension,ratio,saving_img_flag)
                if saving_img_flag:
                    ## produce 64 attn image
                    syn_name_attn=f'{filename_without_extension_forsaving}_{ratio}.jpg'
                    output_image_attn=os.path.join(folder_name_img, syn_name_attn)
                    attn_syn_img=syn_1_jpg(img_64_filename, output_image_attn) #syn 64 attention to 1 image
                    print(img_64_filename)

                if saving_img_flag:
                    ##syn 64 binary attention to  1 image
                    suffix_b = '_binarize'
                    syn_name_binarize=f'{filename_without_extension_forsaving}{suffix_b}_{ratio}.jpg'
                    output_image_binarize=os.path.join(folder_name_img, syn_name_binarize)
                    attn_syn_binarize=syn_1_png(img_64_filename, output_image_binarize)

                ##syn 64 highatom images
                start_time2 = time.time()
                df_frag_smile_single = highlight_chemical_atoms(img_64_filename,filename_without_extension,saving_img_flag,ratio,smiles_address)
                df_frag_smiles_multi=pd.DataFrame(pd.concat([df_frag_smiles_multi,df_frag_smile_single],ignore_index=True))
                end_time2 = time.time()
                print(f"代码2运行时间：{end_time2 - start_time2}秒")

                if saving_img_flag:
                    suffix_c = '_lightatom'
                    syn_name_atom=f'{filename_without_extension_forsaving}{suffix_c}_{ratio}.jpg'
                    output_image_atom=os.path.join(folder_name_img, syn_name_atom)
                    attn_syn_atom=syn_1_jpeg(img_64_filename, output_image_atom)

    ##read criterion
    input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244')
    # frag_smiles_multi_excel=f'df_frag_smiles_multi_find_head_{ratio}.xlsx'  #save functional group path
    # frag_smiles_multi_path= os.path.join(input_files_smiles, frag_smiles_multi_excel)
    # df_frag_smiles_multi.to_excel(frag_smiles_multi_path)

    input_files_smiles_parent=os.path.dirname(os.path.dirname(input_files_smiles))
    df_criterion = pd.read_excel(os.path.join(input_files_smiles_parent, 'USPTO_50K_mark_1000_split_atoms_number_1244_gold.xlsx'))

    df_atoms_error = calculate_reactive_percentage(df_criterion, df_frag_smiles_multi)
    # atoms_error_excel=f'df_atoms_error_find_head_{ratio}.xlsx'  #save functional group path
    # atoms_error_path= os.path.join(input_files_smiles, atoms_error_excel)
    # df_atoms_error.to_excel(atoms_error_path)

    df_single_row=combine_multi_single_column(df_atoms_error)
    # single_row_excel=f'df_single_row_find_head_{ratio}.xlsx'  #save functional group path
    # single_row_path= os.path.join(input_files_smiles, single_row_excel)
    # df_single_row.to_excel(single_row_path)


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
        ratio_multi_excel=f'df_ratio_multi_{ratio}.xlsx'
        df_ratio_multi.to_excel(os.path.join(input_files_smiles, ratio_multi_excel))
        sys.exit()
df_ratio_multi.to_excel(folder_ratio_statistics)







