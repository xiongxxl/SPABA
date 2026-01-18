from remove_arrary_outerside import remove_outer_layer
from tokenizer import tokenize_smiles
import matplotlib.pyplot as plt
import os
from binarize_by_ratio import binarize_by_ratio
import numpy as np

def attn_64_img(attn,folder_name_img,mol_name,ratio,saving_img_flag):
    # input one mol 8*8 attn,
    # output img of the mol attn,jpg is attn ,png is binarize

    # if attn.shape[1]==512:
    #     attn_del1=attn[1:-1]
    #     attn_del1_bin = binarize_by_ratio(attn_del1, ratio)
    #
    #     mol_name = mol_name.replace("/", "x")
    #     file_mol_deep = f'deep_attn_del_{head_axis}'
    #     folder_mol_deep = os.path.join(folder_name_img, file_mol_deep)
    #     if not os.path.exists(folder_mol_deep):
    #         os.makedirs(folder_mol_deep)
    #     mol_deep_npy = f'{mol_name}.npy'
    #     file_path_deep_npy = os.path.join(folder_mol_deep, mol_deep_npy)
    #     np.save(file_path_deep_npy, attn_del1)
    #
    #     file_bin_deep = f'deep_attn_bin_{head_axis}'
    #     folder_bin_deep = os.path.join(folder_name_img, file_bin_deep)
    #     if not os.path.exists(folder_bin_deep):
    #         os.makedirs(folder_bin_deep)
    #     mol_bin_npy = f'{mol_name}.npy'
    #     file_path_bin_npy = os.path.join(folder_bin_deep, mol_bin_npy)
    #     np.save(file_path_bin_npy, attn_del1_bin)
    #
    # else:

    for i in range(8):
        for j in range(8):
            attn_single=attn[i][j]
            attn_del2=remove_outer_layer(attn_single)
            mol_name_suf=f"{mol_name}_{ratio}"
            folder_name_mol=os.path.join(folder_name_img,mol_name_suf)
            if not os.path.exists(folder_name_mol):
                os.makedirs(folder_name_mol)

            if saving_img_flag:
                plt.imshow(attn_del2, cmap='Greys')
                mol_name_ticks=tokenize_smiles(mol_name)
                plt.xticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.yticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.xlabel(f'{i}_{j}')
                single_name_greys = f'{i}_{j}.jpg'
                file_path_greys =os.path.join(folder_name_mol, single_name_greys)
                plt.savefig(file_path_greys)
                plt.close()


            ## decrease ratio for  one mol no highlight atoms
            ratio_decrease_one=ratio-0.01
            path_decrease_one_suf=f"{ratio_decrease_one}"
            folder_decrease_one=os.path.join(folder_name_mol,path_decrease_one_suf)
            if not os.path.exists(folder_decrease_one):
                os.makedirs(folder_decrease_one)

            ## decrease ratio for  one mol no highlight atoms
            ratio_decrease_two=ratio_decrease_one-0.01
            path_decrease_two_suf=f"{ratio_decrease_two}"
            folder_decrease_two=os.path.join(folder_name_mol,path_decrease_two_suf)
            if not os.path.exists(folder_decrease_two):
                os.makedirs(folder_decrease_two)

            if saving_img_flag:
                single_name_greys = f'{i}_{j}.jpg'
                file_path_greys =os.path.join(folder_name_mol, single_name_greys)
                plt.savefig(file_path_greys)
                plt.close()

            ## binarize attn array
            #ratio=0.95 # set front ratio to  0
            attn_del2_bin = binarize_by_ratio(attn_del2,ratio)
            attn_del2_bin_decrease_one = binarize_by_ratio(attn_del2, ratio_decrease_one)
            attn_del2_bin_decrease_two = binarize_by_ratio(attn_del2, ratio_decrease_two)
            # attn_del2_bin = find_and_keep_largest_block(attn_del2_bin) #find max bolk

            if saving_img_flag:
                plt.imshow(attn_del2_bin, cmap='Greys')
                plt.xticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.yticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.xlabel(f'{i}_{j}')
                single_name_binary = f'{i}_{j}.png'
                file_path_binary =os.path.join(folder_name_mol,single_name_binary)
                plt.savefig(file_path_binary)
                plt.close()

            single_name_npy=f'{i}_{j}.npy'
            file_path_single_npy = os.path.join(folder_name_mol,single_name_npy)
            np.save(file_path_single_npy, attn_del2_bin)

            # # saving attention for training
            # if i==4 and j==5:
            #     print('4_5')
            #
            #     mol_name = mol_name.replace("/", "x")
            #     file_mol_deep = f'deep_attn_del_{i}_{j}'
            #     folder_mol_deep = os.path.join(folder_name_img, file_mol_deep)
            #     if not os.path.exists(folder_mol_deep):
            #         os.makedirs(folder_mol_deep)
            #     mol_deep_npy = f'{mol_name}.npy'
            #     file_path_deep_npy=os.path.join(folder_mol_deep,mol_deep_npy)
            #     np.save(file_path_deep_npy,attn_del2)
            #
            #     file_bin_deep = f'deep_attn_bin_{i}_{j}'
            #     folder_bin_deep = os.path.join(folder_name_img, file_bin_deep)
            #     if not os.path.exists(folder_bin_deep):
            #         os.makedirs(folder_bin_deep)
            #     mol_bin_npy = f'{mol_name}.npy'
            #     file_path_bin_npy = os.path.join(folder_bin_deep, mol_bin_npy)
            #     np.save(file_path_bin_npy, attn_del2_bin)


            single_name_npy=f'{i}_{j}.npy'
            file_path_decrease_one_npy = os.path.join(folder_decrease_one,single_name_npy)
            np.save(file_path_decrease_one_npy,attn_del2_bin_decrease_one)

            single_name_npy=f'{i}_{j}.npy'
            file_path_decrease_two_npy = os.path.join(folder_decrease_two,single_name_npy)
            np.save(file_path_decrease_two_npy , attn_del2_bin_decrease_two)

    return folder_name_mol, file_path_decrease_one_npy, file_path_decrease_two_npy
if __name__ == "__main__":
   print()
