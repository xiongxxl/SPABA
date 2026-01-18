import os
import time
import numpy as np
import pandas as pd
from tokenizer import tokenize_smiles
from binary_highatom_combine import highlight_chemical_atoms_single
from syn_1_img import syn_1_jpg
from syn_1_img import syn_1_jpeg
from syn_1_img import syn_1_png
from attn_binary_combine import attn_64_img
from comparison_reactive import calculate_reactive_percentage
from statistic_multi_single import combine_multi_single_column
from combine_two_heads import combine_two_heads
from drop_two_heads import drop_two_heads

j=0
heads=[4_5,7_7]
ratio_adaptive=0.986
saving_img_flag = 0  ## flag=0 is not saving picture,flag=1 is saving picture

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_supervision/uspto_1244/combine')

data_files_npy='data/middle_attention/npy_supervision/multi_1000_v01'
folder_name_npy=os.path.join(parent_dir, data_files_npy)

data_files_img='data/result/img_supervision/multi_1000_v01/combine/64_heads'
folder_name_img=os.path.join(parent_dir, data_files_img)

df_frag_smiles_multi_4_5=pd.DataFrame()
df_frag_smiles_multi_7_7=pd.DataFrame()
df_atoms_number_multi_4_5 =pd.DataFrame()
df_atoms_number_multi_7_7 =pd.DataFrame()

## produce two heads df_atoms_error ###
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
            # data_files_npy_simple='data/middle_attention/npy_reactive/double_molecule_100/NC1=NC=NC=C1N.C(O)=O.npy'
            # attn=np.load(os.path.join(parent_dir, data_files_npy_simple))
            # filename_without_extension='NC1=NC=NC=C1N.C(O)=O'
            smiles_address=filename_without_extension  #this filename for saving address
            filename_without_extension = filename_without_extension.replace("x", "/")
            filename_without_extension_re= tokenize_smiles(filename_without_extension)  #re syn like cl to one element
            j=j+1
            print(f'Filename: {filename_without_extension}')
            print(j)

            start_time1=time.time()
            img_64_filename=attn_64_img(attn,folder_name_img,filename_without_extension,ratio,saving_img_flag) #produce 64 attn image

            if saving_img_flag:
                syn_name_attn=f'{smiles_address}_{ratio}.jpg'
                output_image_attn=os.path.join(folder_name_img, syn_name_attn)
                attn_syn_img=syn_1_jpg(img_64_filename, output_image_attn) #syn 64 attention to 1 image
                print(img_64_filename)

            end_time1=time.time()
            print(f"代码1运行时间：{end_time1-start_time1}秒")

            if saving_img_flag:    #syn 64 binary attention to  1 image
                suffix_b = '_binarize'
                syn_name_binarize=f'{smiles_address}{suffix_b}_{ratio}.jpg'
                output_image_binarize=os.path.join(folder_name_img, syn_name_binarize)
                attn_syn_binarize=syn_1_png(img_64_filename, output_image_binarize)

            ##syn 64 highatom images
            start_time2 = time.time()
            df_frag_smile_single_4_5, df_frag_smile_single_7_7 = highlight_chemical_atoms_single(img_64_filename,
                                                        filename_without_extension,saving_img_flag,ratio,smiles_address)
            df_frag_smiles_multi_4_5=pd.DataFrame(pd.concat([df_frag_smiles_multi_4_5,df_frag_smile_single_4_5],ignore_index=True))
            df_frag_smiles_multi_7_7 = pd.DataFrame(pd.concat([df_frag_smiles_multi_7_7, df_frag_smile_single_7_7], ignore_index=True))
            end_time2 = time.time()
            print(f"代码2运行时间：{end_time2 - start_time2}秒")

            if saving_img_flag:     #syn 64  highlight to  1 image
                suffix_c = '_lightatom'
                syn_name_atom=f'{smiles_address}{suffix_c}_{ratio}.jpg'
                output_image_atom=os.path.join(folder_name_img, syn_name_atom)
                attn_syn_atom=syn_1_jpeg(img_64_filename, output_image_atom)

df_frag_smiles_multi_4_5.to_excel(os.path.join(input_files_smiles, 'df_frag_smiles_multi_4_5.xlsx'))
df_frag_smiles_multi_7_7.to_excel(os.path.join(input_files_smiles, 'df_frag_smiles_multi_7_7.xlsx'))

criterion_path=os.path.dirname(input_files_smiles)
df_criterion = pd.read_excel(os.path.join(criterion_path, 'multi_criterion_1000_del_v01_20250221.xlsx'))

df_atoms_error_4_5 = calculate_reactive_percentage(df_criterion, df_frag_smiles_multi_4_5)
df_single_row_4_5=combine_multi_single_column(df_atoms_error_4_5)
df_single_row_4_5.to_excel(os.path.join(input_files_smiles, 'df_single_row_4_5.xlsx'))
df_atoms_error_4_5.to_excel(os.path.join(input_files_smiles, 'df_atoms_error_4_5.xlsx'))

df_atoms_error_7_7 = calculate_reactive_percentage(df_criterion, df_frag_smiles_multi_7_7)
df_single_row_7_7=combine_multi_single_column(df_atoms_error_7_7)
df_atoms_error_7_7.to_excel(os.path.join(input_files_smiles, 'df_atoms_error_7_7.xlsx'))
df_single_row_7_7.to_excel(os.path.join(input_files_smiles, 'df_single_row_7_7.xlsx'))

### combine two heads ###
src_folder_img=combine_two_heads()

### combine and drop two heads ###
merged_df=drop_two_heads()






























