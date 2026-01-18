from remove_arrary_outerside import remove_outer_layer
from tokenizer import tokenize_smiles
import matplotlib.pyplot as plt
import os
from binarize_by_ratio import binarize_by_ratio
import numpy as np

def attn_64_img(attn,folder_name_img,mol_name,ratio,saving_img_flag):
    # input one mol 8*8 attn,
    # output img of the mol attn,jpg is attn ,png is binarize
    for i in range(8):
        for j in range(8):

            attn_single=attn[i][j]
            attn_del2=remove_outer_layer(attn_single)

            if saving_img_flag:
                plt.imshow(attn_del2, cmap='Greys')
                mol_name_ticks=tokenize_smiles(mol_name)
                plt.xticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.yticks(range(len(mol_name_ticks)),mol_name_ticks)
                plt.xlabel(f'{i}_{j}')

            mol_name_suf=f"{mol_name}_{ratio}"
            folder_name_mol=os.path.join(folder_name_img,mol_name_suf)
            if not os.path.exists(folder_name_mol):
                os.makedirs(folder_name_mol)

            if saving_img_flag:
                single_name_greys = f'{i}_{j}.jpg'
                file_path_greys =os.path.join(folder_name_mol, single_name_greys)
                plt.savefig(file_path_greys)
                plt.close()

            attn_del2_bin = binarize_by_ratio(attn_del2,ratio)   ## binarize attn array

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

    return folder_name_mol

if __name__ == "__main__":
   print()
