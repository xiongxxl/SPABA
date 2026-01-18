import pandas as pd
# 示例数据
import pandas as pd
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

import torch
import pandas as pd
import numpy as np
import hashlib
from ast import literal_eval
import itertools
from pathlib import Path
import pandas as pd
import os
from rdkit import Chem




def remove_outer_row(matrix):
    """去掉矩阵的最外1行（首行和末行）"""
    return matrix[1:-1]
# 示例
matrix = [
    [1, 2, 3],   # 第1行（将被移除）
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12], # 最后1行（将被移除）
]
result = remove_outer_row(matrix)
print("处理后的矩阵:")
for row in result:
    print(row)











# # 假设你的Excel文件路径如下
# excel_file_1 = 'file1.xlsx'
# excel_file_2 = 'file2.xlsx'
# # 初始化一个空的 DataFrame
# merged_df = pd.DataFrame()
# # 判断第一个文件是否存在
# if os.path.exists(excel_file_1) and os.path.exists(excel_file_2):
#     # 如果两个文件都存在，读取两个文件并去重
#     df1 = pd.read_excel(excel_file_1)
#     df2 = pd.read_excel(excel_file_2)
#     merged_df = pd.concat([df1, df2]).drop_duplicates()  # 合并并去重
# elif os.path.exists(excel_file_1):
#     # 如果只有第一个文件存在
#     merged_df = pd.read_excel(excel_file_1)
# elif os.path.exists(excel_file_2):
#     # 如果只有第二个文件存在
#     merged_df = pd.read_excel(excel_file_2)
# # 打印合并后的 DataFrame
# print(merged_df)




# reactive_atoms_4_5=np.NaN
# reactive_atoms_7_7=[1,3]
# reactive_atoms_4_5_contains_nan=any(pd.isna(reactive_atoms_4_5))
# if pd.isna(reactive_atoms_4_5)  and not pd.isna(reactive_atoms_7_7):
#    print('ok')


import pandas as pd
# 假设A和B是两个列表
# # A = [1.0,float('nan'), 3.0] #A中包含NaN
# A=[1,2]
# # A=[]
# B = [1.0, 2.0, 3.0]           # B中不包含NaN
# # 判断A中是否包含NaN值
# # A_contains_nan = any(pd.isna(val) for val in A)
# # # 判断B中是否不包含NaN值
# # B_contains_no_nan = all(not pd.isna(val) for val in B)
# # 结合判断
# if len(A)==0 and len(B)>0:
#     print("A中包含NaN，B中不包含NaN")
# else:
#     print("条件不满足")


 #ATOM_TOKENS = [
#     'c', 'C', 'O', 'N', 'n', '[C@@H]',
#     '[C@H]', 'F', '[NH+]', 'S', 's', 'Cl', 'o', '[nH]', '[NH2+]',
#     '[nH+]', '[O-]', '#', 'Br', '[NH3+]', '[C@@]', '[C@]',
#     '[N-]', '[n-]', 'I', '[N+]', '[S@@]', '[S@]', '[N]',
#     '[H]', '[NH]', '[NH-]', '[C]', '[S+]', '[n+]', '[CH]', 'P',
#     '[O+]', '[o+]', '[O]', '[NH2]', '[CH-]', '[P]',
#     '[Si]', '[C-]', '[s+]', '[OH+]', '[2H]', '[NH3]', '[N@@]',
#     '[CH2-]', '[C@@H-]', '[P+]', '[S-]', '[As]', '[N@]', '[Br-]',
#     '[Sn]', '[CH2]', '[I-]', '[Hg]', 'B', '[SH]', '[PH]',
#     '[Cl-]', '[S]', '[B-]', '[AlH3]', '[Se]', '[C@H-]', '[N@@+]',
#     '[S@+]', '[N@+]', '[Pt]', '[se]', '[cH-]', '[P@]', '[N@@H]',
#     '[S@@+]', '[Cr]', '[Cl+3]', '[I+]', '[Cu]', '[P@@]', '[Zn+2]',
#     '[c+]', '[Na+]', '[Te]', '[Fe]', '[c-]', '[IH2]', '[Ba+2]', '[Cd]',
#     '[Au+]', '[Zn]', '[F]', '[Bi]', '[Sb]', '[Mo]', '[Cu+2]',
#     'p', '[N@H]', '[Cl]', '[Au]', '[In]', '[Pt+2]', '[Pd]', '[B]',
#     '[N@H+]', '[3H]', '[NH4+]', '[Ca+2]', '[K+]', '[Hg+2]', '[Fe+2]',
#     '[Co+2]', '[Fe+3]', '[SiH]', '[Ge]', '[Ag]', '[SH-]', '[Co]',
#     '[Cu+]', '[V]', '[C@-]', '[13C]', '[H+]', '[Mg+2]', '[Zr]', '[Na]',
#     '[Gd+3]', '[Co+]', '[Li+]', '[Ni+2]', '[Mn+2]', '[Ti]', '[Ni]',
#     '[GaH3]', '[Ac]', '[BiH3]', '[Mo-2]', '[pH]', '[N++]', '[Br]',
#     '[15N]', '[OH-]', '[TlH2+]', '[Ba]', '[Ag+]', '[Cr+3]', '[Nd]',
#     '[Yb]', '[PbH2+2]', '[Cd+2]', '[SnH2+2]', '[Ti+2]', '[Dy]', '[Ca]',
#     '[Sr+2]', '[Be+2]', '[Cr+2]', '[Mn+]', '[SbH6+3]', '[Au-]', '[Fe-]',
#     '[Fe-2]',
#     '[U]'
# ]
#
# def is_atom(token):
#     # 检查是否是单个原子符号
#     if token in ATOM_TOKENS:
#         return True
#     return False
#
# def get_bonds_mat(seq):
#     """
#     this method extracts the bond matrix for a sequence using RDKIT,
#     with improved atom identification
#     """
#     mol = Chem.MolFromSmiles(seq)
#     bonds = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol.GetBonds()]
#     bonds_mat = np.zeros((len(mol.GetAtoms()), len(mol.GetAtoms())))
#
#     for ele in bonds:
#         bonds_mat[ele[0], ele[1]] = 1
#         bonds_mat[ele[1], ele[0]] = 1
#
#     bond_tokens = []
#
#     for atom in mol.GetAtoms():
#         # 获取原子的SMILES表示
#         atom_symbol = atom.GetSmarts()
#         # 如果是普通原子(单个字母),确保大写
#         if len(atom_symbol) == 1:
#             atom_symbol = atom_symbol.upper()
#         bond_tokens.append(atom_symbol)
#
#     # 过滤出真正的原子
#     real_atoms = []
#     real_bond_mat = []
#     atom_idx_map = {}  # 用于映射原始索引到新索引
#
#     new_idx = 0
#     for idx, token in enumerate(bond_tokens):
#         if is_atom(token):
#             real_atoms.append(token)
#             atom_idx_map[idx] = new_idx
#             new_idx += 1
#
#     # 重建键矩阵
#     n = len(real_atoms)
#     real_bond_mat = np.zeros((n, n))
#
#     # 使用映射更新键矩阵
#     for i in range(len(bond_tokens)):
#         for j in range(len(bond_tokens)):
#             if bonds_mat[i, j] == 1:
#                 if i in atom_idx_map and j in atom_idx_map:
#                     new_i = atom_idx_map[i]
#                     new_j = atom_idx_map[j]
#                     real_bond_mat[new_i, new_j] = 1
#
#     return real_bond_mat, real_atoms
#
#
# def get_attention_matrix(sequence):
#     """获取attention矩阵并转换为numpy格式"""
#     attentions, tokens = get_full_attention(sequence)
#
#     # 获取维度信息
#     num_layers = len(attentions)
#     seq_len = attentions[0].size(-1)
#
#     # 创建numpy数组
#     attention_array = np.zeros((num_layers, seq_len, seq_len))
#
#     # 将每层的attention矩阵转换为numpy
#     for i, layer_attention in enumerate(attentions):
#         layer_numpy = layer_attention.cpu().numpy()
#         attention_array[i] = layer_numpy.squeeze(0).mean(axis=0)
#
#     return attention_array, tokens
#
# if __name__ == "__main__":
#     smiles='C=CC1=CC=CC=C1.O=CCC'
#     real_bond_mat, real_atoms=get_bonds_mat(smiles)
#     A=real_bond_mat





# 假设 A、B、C 三个列表如下
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
#
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# # 输入列表
# input_list = [8]
# # 生成结果
# output_list = [[input_list[i] - 1, input_list[i]] for i in range(len(input_list))] + [[input_list[i], input_list[i] + 1] for i in range(len(input_list))]
# print(output_list)
#
# def standardize_single_smiles(smiles):
#     """
#     将单个 SMILES 转换为标准化 SMILES，无法解析时返回原始 SMILES。
#     Args:
#         smiles (str): 输入的 SMILES 字符串。
#     Returns:
#         str: 标准化后的 SMILES，或者原始 SMILES（如果无法解析）。
#     """
#     try:
#         # 转换为分子对象
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             # 转换为标准化的 SMILES
#             return Chem.MolToSmiles(mol, canonical=True)
#         else:
#             # 如果无法解析，返回原始 SMILES
#             return smiles
#     except Exception as e:
#         # 捕获异常并返回原始 SMILES
#         print(f"Error processing SMILES {smiles}: {e}")
#         return smiles
# # 示例 SMILES
# # raw_smiles = "C1=CC=CC=C1"  # 这是苯的SMILES
# # standardized_smiles = standardize_single_smiles(raw_smiles)
# # print(f"Input SMILES: {raw_smiles}")
# # print(f"Standardized SMILES: {standardized_smiles}")
#
# # # 示例 SMILES 输入
# # input_smiles = ["C1=CC=CC=C1", "CC(C)C(=O)O", "C#CCBr", "INVALID_SMILES"]
# # output_smiles = standardize_smiles(input_smiles)
#
#
# def contains_benzene(smiles):
#     # 将SMILES转换为分子对象
#     smiles=standardize_single_smiles(smiles)
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return False  # 如果SMILES无效，则返回False
#
#     # 遍历分子中的所有环
#     for ring in mol.GetRingInfo().AtomRings():
#         # 判断环的大小是否为6，并且包含交替双键（芳香性）
#         if len(ring) == 6:
#             # 获取环中所有的原子
#             atoms_in_ring = [mol.GetAtomWithIdx(i) for i in ring]
#             # 检查是否为芳香性环（即苯环）
#             if all(atom.GetIsAromatic() for atom in atoms_in_ring):
#                 return True
#     return False
#
#
# # # 测试样例
# # smiles_1 = "C1=CC=CC=C1"  # 苯环
# # smiles_2 = "N#CC1=CC=CS1.OC"  # 不含苯环
# # print(contains_benzene(smiles_1))  # 输出：True
# # print(contains_benzene(smiles_2))  # 输出：False
#
#
# def remove_fragment_indices(smiles,frag_indices):
#     smiles = standardize_single_smiles(smiles)
#     if contains_benzene(smiles):
#         mol = Chem.MolFromSmiles(smiles)
#         benzene_ring = Chem.MolFromSmiles('c1ccccc1')
#         #find benzene location
#         benzene_matches =list(mol.GetSubstructMatches(benzene_ring)[0])
#         frag_indices_del_fragment= [x for x in frag_indices if x not in benzene_matches]
#
#     else:
#         frag_indices_del_fragment=frag_indices
#
#     return frag_indices_del_fragment
#
#
#
#
# location_restrain=[]
# smiles="C=CC1=CC=CC=C1.O=CCC"
# location=[4,5]
# reactive_atoms=remove_fragment_indices(smiles,location)
#
# print(reactive_atoms)
#


#def standardize_single_smiles(smiles):
#     """
#     将单个 SMILES 转换为标准化 SMILES，无法解析时返回原始 SMILES。
#     Args:
#         smiles (str): 输入的 SMILES 字符串。
#     Returns:
#         str: 标准化后的 SMILES，或者原始 SMILES（如果无法解析）。
#     """
#     try:
#         # 转换为分子对象
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             # 转换为标准化的 SMILES
#             return Chem.MolToSmiles(mol, canonical=True)
#         else:
#             # 如果无法解析，返回原始 SMILES
#             return smiles
#     except Exception as e:
#         # 捕获异常并返回原始 SMILES
#         print(f"Error processing SMILES {smiles}: {e}")
#         return smiles

# 示例 SMILES
# raw_smiles = "C1=CC=CC=C1"  # 这是苯的SMILES
# standardized_smiles = standardize_single_smiles(raw_smiles)
# print(f"Input SMILES: {raw_smiles}")
# print(f"Standardized SMILES: {standardized_smiles}")

# # 示例 SMILES 输入
# input_smiles = ["C1=CC=CC=C1", "CC(C)C(=O)O", "C#CCBr", "INVALID_SMILES"]
# output_smiles = standardize_smiles(input_smiles)


from rdkit import Chem
# import os
# import pandas as pd
# import pandas as pd
# # 创建一个示例 DataFrame
# # data = {'A': [1, 2, 3, 1],
# #         'B': [4, 1, 6, 7],
# #         'C': [1, 1, 9, 10]}
# # df = pd.DataFrame(data)
# # # 删除包含1的行
# # df_cleaned = df[~df.isin([1]).any(axis=1)]
# # print(df_cleaned)


# def remove_duplication_flag(df):
#     # 假设你有一个 DataFrame df
#     # 创建一个示例 DataFrame
#     # data = {'flag_0': [1, 0, 1, 0, 1, 0],
#     #         'flag_1': [0, 1, 1, 0, 1, 0],
#     #         'flag_2': [1, 1, 0, 1, 0, 1],
#     #         'flag_3': [0, 0, 1, 0, 1, 1],
#     #         'flag_4': [1, 0, 1, 1, 0, 0],
#     #         'flag_5': [0, 1, 0, 1, 0, 1]}
#     # df = pd.DataFrame(data)
#     # 用于存储每个flag的统计结果
#     counts = {}
#     # 依次统计flag_0到flag_5中为1的个数
#     for i in range(6):  # 从flag_0到flag_5
#         flag_column = f'flag_{i}'
#         flag_column='flag_1'
#
#         df=df['flag_1'].str.replace("]","").str.replace("[","")
#
#
#         # 删除当前flag为1的行
#         df = df[df[flag_column] != 1]
#     # 将结果转换为Series
#     counts_series = pd.Series(counts)
#     # 输出最终的Series
#     print(counts_series)
#     return counts_series
#
# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
# statistics_detail = 'data/result/statistics_reactive/double_100'
# statistics_folder = os.path.join(parent_dir, statistics_detail)
# atoms_number_excel = f'frag_atoms_error_main_0.9.xlsx'  # save functional group path
# statistics_atoms_number_path = os.path.join(statistics_folder, atoms_number_excel)
# df_atoms_error = pd.read_excel(statistics_atoms_number_path)
# df_single_row = remove_duplication_flag(df_atoms_error)
# print(df_single_row)

# def standardize_smiles(smiles_list):
#     """
#     将 SMILES 转换为标准化形式，无法解析时返回原始 SMILES。
#
#     Args:
#         smiles_list (list of str): 输入 SMILES 字符串的列表。
#     Returns:
#         list of str: 转换后的标准化 SMILES 列表，无法解析时保留原始 SMILES。
#     """
#     standardized = []
#     for smiles in smiles_list:
#         try:
#             # 转化为分子对象
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 # 转换为标准化的 SMILES 表示
#                 standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
#                 standardized.append(standardized_smiles)
#             else:
#                 # 无法解析的 SMILES，返回原始 SMILES
#                 standardized.append(smiles)
#         except Exception as e:
#             # 出现错误时也返回原始 SMILES
#             print(f"Error processing SMILES {smiles}: {e}")
#             standardized.append(smiles)
#     return standardized
#
#
# # 示例 SMILES 输入
# input_smiles = ["C1=CC=CC=C1", "CC(C)C(=O)O", "C#CCBr", "INVALID_SMILES"]
# output_smiles = standardize_smiles(input_smiles)
#
# # 打印结果
# for i, (raw, std) in enumerate(zip(input_smiles, output_smiles)):
#     print(f"Input SMILES: {raw}, Standardized SMILES: {std}")



# data = [('C=C', (2, 3)),
#         ('C=C', (4, 5)),
#         ('R-OH', (0,)),
#         ('R-OH', (8,)),
#         ('R-CO-R', (1, 7, 2)),
#         ('R-COOH', (1, 7, 0))]
# if any(item[0] == 'R-COOH' for item in data) and any(item[0] == 'R-CO-R' for item in data):
#     data = [item for item in data if item[0] != 'R-CO-R']
# print(data)


# A = [9, 8, 10]
# B = [ [7,8],[8,9]]
# # 遍历B中的每个子列表
# for sublist in B:
#     # 判断A中是否包含子列表的元素
#     if all(item in A for item in sublist):
#         # 如果包含，将该子列表赋值给A
#         A = sublist
#
# print(A)





#
# # 官能团字典
# functional_groups = {
#     "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#     "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#     "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#     "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#     "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#     "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
#     "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#     "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#     "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#     "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#     "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#     "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#     "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#     "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#     "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#     "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#     "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#     "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#     "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#     "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#     "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#     "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
#     "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
# }
#
#
# # 分子识别功能团并输出原子索引
# def identify_functional_groups_with_indices(molecule_smiles):
#     # 将SMILES转化为分子对象
#     mol = Chem.MolFromSmiles(molecule_smiles)
#     if not mol:
#         return "Invalid SMILES"
#
#     identified_groups = []
#
#     # 遍历官能团字典中的每个功能团，尝试匹配
#     for group_name, group_data in functional_groups.items():
#         smarts_pattern = group_data['smarts']
#         pattern = Chem.MolFromSmarts(smarts_pattern)
#
#         # 获取匹配的原子索引
#         matches = mol.GetSubstructMatches(pattern)
#
#         # 如果有匹配，记录下来
#         if matches:
#             for match in matches:
#                 # 转换原子索引为实际原子编号
#                 atom_indices = [mol.GetAtomWithIdx(idx).GetIdx() for idx in match]
#                 identified_groups.append((group_name, atom_indices))
#
#     return identified_groups
#
#
# # 示例分子
# molecule_smiles = "NCC1=CC=CC=C1.O=CCCCCCCC"
# identified_groups = identify_functional_groups_with_indices(molecule_smiles)
#
# if identified_groups:
#     for group, indices in identified_groups:
#         print(f"Functional group: {group}, Atom indices: {indices}")
# else:
#     print("No functional groups found.")
# # 官能团字典
# functional_groups = {
#     "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#     "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#     "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#     "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#     "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#     "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
#     "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#     "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#     "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#     "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#     "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#     "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#     "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#     "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#     "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#     "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#     "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#     "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#     "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#     "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#     "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#     "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
#     "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
# }
#
#
# # 分子识别功能团
# def identify_functional_groups(molecule_smiles):
#     # 将SMILES转化为分子对象
#     mol = Chem.MolFromSmiles(molecule_smiles)
#     if not mol:
#         return "Invalid SMILES"
#
#     identified_groups = []
#
#     # 遍历官能团字典中的每个功能团，尝试匹配
#     for group_name, group_data in functional_groups.items():
#         smarts_pattern = group_data['smarts']
#         pattern = Chem.MolFromSmarts(smarts_pattern)
#
#         # 如果模式匹配，则记录
#         if mol.HasSubstructMatch(pattern):
#             identified_groups.append(group_name)
#
#     return identified_groups
#
#
# # 示例分子
# molecule_smiles = "NCC1=CC=CC=C1.O=CCCCCCCC"
# identified_groups = identify_functional_groups(molecule_smiles)
#
# print(f"The functional groups in the molecule are: {identified_groups}")


# # A = ['1', '3',  '8']
# # B = ['1', '2', '5']
# # C = ['6', '7', '8']
# A=[0,8,1,7,2,17,0]
# B = ['1','2','3','4','5','6','7','8']
# C = ['9','10','11']
# F=eval(C)
# D=set(A).intersection(B)
# E=set(A).intersection(F)
# # 判断 A 中是否既有 B 的元素，又有 C 的元素
# if set(A).intersection(B) and set(A).intersection(C):
#     result = True
# else:
#     result = False
# # 输出结果
# print(result)






# import sys
# def main():
#     i = 0
#     while i < 5:
#         print(i)
#         i += 1
#     sys.exit()
# main()




# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# # 生成随机数据
# x = np.random.rand(50)
# y = np.random.rand(50)
# z = np.random.rand(50)
# # 创建图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 绘制3D散点图
# ax.scatter(x, y, z, c='r', marker='o')
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# # 显示图形
# plt.show()



# import pandas as pd
# import re
# from collections import Counter
# # 创建示例 DataFrame
# data = {'column': ["'0_7''4_0''1_5' '5_0' '2_6''0_5''5_4''1_7''1_6''0_7''3_0''0_7''0_7''3_4''5_0''4_0''5_5''7_4''1_3' '0_0' '1_0''5_5''7_0' '7_6''6_7'"]}
# df = pd.DataFrame(data)
# # 提取数字和下划线的函数
# def extract_numbers(text):
#     return re.findall(r'\d+_\d+', text)
# # 应用函数提取数字和下划线
# df['extracted'] = df['column'].apply(extract_numbers)
# # 将提取的数字列表展开为一个单一列表
# all_numbers = [item for sublist in df['extracted'] for item in sublist]
# # 使用 Counter 统计每个数字出现的次数
# counter = Counter(all_numbers)
# # 获取出现次数最多的前5个数字
# top_5 = counter.most_common(5)
# # 将前5个数字及其出现次数转换为字符串格式，例如 "0_7: 6, 5_0: 4"
# top_5_str = ', '.join([f"{num}: {count}" for num, count in top_5])
# # 将合成的字符串写入新的 'rank' 列
# df['rank'] = [top_5_str] * len(df)  # 将同样的字符串应用到所有行
# # 显示 DataFrame
# print(df[['column', 'rank']])


# Create a sample DataFrame
# data = {'column': ["'0_7''4_0''1_5' '5_0' '2_6''0_5''5_4''1_7''1_6''0_7''3_0''0_7''0_7''3_4''5_0''4_0''5_5''7_4''1_3' '0_0' '1_0''5_5''7_0' '7_6''6_7'"]}
# df = pd.DataFrame(data)
# # Define a function to extract the numbers and underscores from a string
# def extract_numbers(text):
#     return re.findall(r'\d+_\d+', text)
# # Apply the function to the column in the DataFrame
# df['extracted'] = df['column'].apply(extract_numbers)
# # Flatten the lists of numbers into a single list
# all_numbers = [item for sublist in df['extracted'] for item in sublist]
# # Use Counter to count occurrences of each number
# counter = Counter(all_numbers)
# # Get the top 5 most common numbers
# top_5 = counter.most_common(5)
# # Prepare the result in a single row format
# top_5_row = dict(top_5)  # Convert to dictionary, where keys are numbers and values are counts
# # Convert to DataFrame (columns are the numbers, values are their counts)
# top_5_df = pd.DataFrame([top_5_row])
# print(top_5_df)

# import pandas as pd
# import re
# from collections import Counter
# # Create a sample DataFrame
# data = {'column': ["'0_7''4_0''1_5' '5_0' '2_6''0_5''5_4''1_7''1_6''0_7''3_0''0_7''0_7''3_4''5_0''4_0''5_5''7_4''1_3' '0_0' '1_0''5_5''7_0' '7_6''6_7'"]}
# df = pd.DataFrame(data)
# # Define a function to extract the numbers and underscores from a string
# def extract_numbers(text):
#     return re.findall(r'\d+_\d+', text)
# # Apply the function to the column in the DataFrame
# df['extracted'] = df['column'].apply(extract_numbers)
# # Flatten the lists of numbers into a single list
# all_numbers = [item for sublist in df['extracted'] for item in sublist]
# # Use Counter to count occurrences of each number
# counter = Counter(all_numbers)
# # Get the top 5 most common numbers
# top_5 = counter.most_common(5)
# # Display the top 5
# print("Top 5 most frequent numbers:")
# for rank, (number, count) in enumerate(top_5, start=1):
#     print(f"Rank {rank}: {number} (Count: {count})")


# Create a sample DataFrame
# data = {'column': ["'0_7''4_0''1_5' '5_0' '2_6''0_5''5_4''1_7''1_6''0_7''3_0''0_7''0_7''3_4''5_0''4_0''5_5''7_4''1_3' '0_0' '1_0''5_5''7_0' '7_6''6_7'"]}
# df = pd.DataFrame(data)
# # Define a function to extract the numbers and underscores from a string
# def extract_numbers(text):
#     return re.findall(r'\d+_\d+', text)
# # Apply the function to the column in the DataFrame
# df['extracted'] = df['column'].apply(extract_numbers)
# # Display the DataFrame with the new column
# print(df)


# data = {'head_combined': ['abc_123', 'def_456', 'ghi_789', 'jkl_123', 'abc_123', 'ghi_789', 'abc_123']}
# df_single_row = pd.DataFrame(data)
# # 统计每个元素的出现次数，并选出前 3 个
# top_3 = df_single_row['head_combined'].value_counts().head(3)
# # 将排名前 3 的元素及其计数合并为一个单元格
# top_3_str = ''.join([f"{item}: {count}" for item, count in top_3.items()])
# # 输出合并后的结果
# print(top_3_str)



#
#
# import pandas as pd
# import os
# # 示例数据
# data = {
#     'flag_0': [1, 0, 1, 1],
#     'flag_1': [0, 1, 0, 1],
#     'flag_2': [1, 1, 1, 0],
#     'flag_3': [0, 0, 1, 1],
#     'flag_4': [1, 0, 1, 0],
#     'flag_5': [0, 1, 1, 0],
#     'head_axis_0': ['A', 'B', 'C', 'D'],
#     'head_axis_1': ['E', 'F', 'G', 'H'],
#     'head_axis_2': ['I', 'J', 'K', 'L'],
#     'head_axis_3': ['M', 'N', 'O', 'P']
#  }
# # # 创建 DataFrame
# # current_dir = os.getcwd()
# # parent_dir = os.path.dirname(current_dir)
# # statistics_detail = 'data/result/statistics_reactive/double_100'
# # statistics_folder = os.path.join(parent_dir, statistics_detail)
# # atoms_number_excel = f'frag_atoms_error_main_0.9.xlsx'  # save functional group path
# # statistics_atoms_number_path = os.path.join(statistics_folder, atoms_number_excel)
# # df_atoms_error = pd.read_excel(statistics_atoms_number_path)
# #
# #
# df = pd.DataFrame(data)
# # 计算 flag_0 到 flag_5 列为 1 的个数
# flag_columns = [f'flag_{i}' for i in range(6)]
# flag_count = (df[flag_columns] == 1).sum(axis=0)
# # 合并 head_axis_0 到 head_axis_3 列的数据
# head_columns = [f'head_axis_{i}' for i in range(4)]
# head_combined = df[head_columns].apply(lambda x: ','.join(x), axis=1)
# # 生成结果
# result = pd.DataFrame([flag_count.tolist() + [','.join(head_combined)]], columns=['flag_0', 'flag_1', 'flag_2', 'flag_3', 'flag_4', 'flag_5', 'head_combined'])
# result.to_excel('result.xlsx')
# print(result['head_combined'])
#
# # import pandas as pd
# # import re
# # # 读取 Excel 文件中的数据，假设文件路径为 'your_file.xlsx'，并且目标列是 'head_combined'
# # df = pd.read_excel('result.xlsx')
# # # 处理 'head_combined' 列数据
# # df['head_combined'] = df['head_combined'].apply(lambda x: re.sub(r'\[\]', '', str(x)))  # 去掉空的 []
# # df['head_combined'].replace({',':''}, regex=True)
# # # 对有值的 [] 拆开，逗号分隔的值替换为空格
# # df['head_combined'] = df['head_combined'].apply(lambda x: re.sub(r'\[([^\]]+)\]', lambda m: m.group(1).replace(',', ' ').strip(), str(x)))
# # # 将处理后的 DataFrame 保存回 Excel 文件（可选）
# # df.to_excel('processed_file.xlsx', index=False)
# # # 如果想查看处理结果
# # print(df.head())
#
#
#


# from collections import Counter
# # 定义列表
# data = ["['1_0']", "['2_3']", "['2_3']", "['2_1']", "['4_3']"]
# # 统计每个元素的出现次数
# counter = Counter(data)
# # 获取出现次数最多的前三个元素
# top_three = counter.most_common(3)
# # 输出结果
# for rank, (element, count) in enumerate(top_three, start=1):
#     print(f"排名 {rank}: 元素 {element}，出现次数 {count}")






#
# original_list = ['1_0']['2_3']['2_3']['2_1','1_5']
# # 将原始列表转换为标准Python列表
# original_list = eval(original_list)
# 分割列表
# 由于你的列表格式有点特殊，我们先将其转换为标准列表格式
# 假设我们按照你提供的格式，将列表分割为两部分
# 第一部分是['1_0', '2_3', '2_3']
# 第二部分是['2_1', '1_5']
# part1 = original_list[:3]
# part2 = original_list[3:]
# # 对第二部分的元素进行排名
# # 排名按照元素在原始列表中出现的顺序
# ranked_elements = {element: i + 1 for i, element in enumerate(part2)}
# # 输出结果
# print("分割后的第一部分:", part1)
# print("分割后的第二部分:", part2)
# print("第二部分元素排名:", ranked_elements)
#


# import pandas as pd
# # 创建一个包含空数组的 Series
# data = [ [1, 2], [], [3, 4], [], [5] ]
# series = pd.Series(data)
# # 使用apply()来检查每个元素是否为空数组
# A=series.apply(lambda x: len(x) > 0)
# cleaned_series = series[A]
# # 输出结果
# print(cleaned_series)




# fuctional_arrays=tuple
#
# for fucntional_array in functional_arrays:
#     if set(location_del_benzene)&set(fucntional_array):
#         location_restrain.extend(fucntional_array)
# print('location_restrain',location_restrain)

# import numpy as np
# # 假设有两个一维的 ndarray 数组
# array1 = np.array([1, 2, 3, 4, 5])
# array2 = np.array([4, 5, 6, 7, 8])
# # 合并数组
# combined_array = np.concatenate((array1, array2))
# # 去重
# unique_array = np.unique(combined_array)
# print(unique_array)
#
#
# #
# A=pd.read_excel('/mnt/work/code/tian/smiles/data/result/statistics_reactive/double_100/double_100_criterion.xlsx')
# #
# A = (str([4,3,5,6,7,12]))
# B = str([3,2,4,5,6,11])
# A=eval([4,3,5,6,7,12])
# B=eval(B)
# print(sorted(A))
# print(sorted(B))
#
# def get_diff(A,B):
#     diff = -1
#     for b in B:
#         if b in A:
#             pass
#         else:
#             return diff
#     diff = len(A) - len(B)
#     return diff
#
# print(get_diff(A,B))

#
# def find_amino_nitrogen_index(smiles):
#     # SMILES转化为分子对象
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 定义Amine官能团的SMARTS模式
#     amine_smarts = "[NX3][#6]"  # R-NH2的SMARTS模式，匹配含氮的氨基
#
#     # 构建Amine的SMARTS模式
#     amine_pattern = Chem.MolFromSmarts(amine_smarts)
#
#     # 查找Amine官能团的匹配项
#     matches = mol.GetSubstructMatches(amine_pattern)
#
# #     # 仅保留氨基中的氮原子索引
# #     nitrogen_indices = [match[0] for match in matches]  # 只取每个匹配中的氮原子索引
# #
# #     return nitrogen_indices
# #
# #
# # # 测试分子
# # smiles = "FC1=C(Br)C=C(C=O)C=C1.NCC(OC)OC"
# # nitrogen_indices = find_amino_nitrogen_index(smiles)
# # # 输出结果
# # print("Amine (R-NH2) Nitrogen Indices:", nitrogen_indices)
# # 多个元组
# data = [
#     ('X (F, Cl, Br, I)', (12,)),
#     ('R-O-R', (8, 7, 9)),
#     ('R-COO-R', (9, 10, 8, 7)),
#     ('R-O-R', (3, 2, 5))
# ]
# # 处理每个元组，去掉 'R-O-R' 中的最小值
# updated_data = [
#     (item[0], tuple(x for x in item[1] if x != min(item[1]))) if item[0] == 'R-O-R' else item
#     for item in data
# ]
# # 输出结果
# print(updated_data)
#
#
# def find_functional_axis(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 官能团集合
#     functional_groups = {
#         "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#         "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#         "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#         "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#         "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#         # 恢复原SMARTS模式，确保能够匹配酮基
#         "Ketone": {"smarts": "[CX3](=O)[CX3]", "formula": "R-CO-R"},
#         "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#         "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#         "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#         "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#         "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#         "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#         "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#         "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#         "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#         "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#         "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#         "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#         "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#         "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#         "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#         "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
#         "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
#     }
#
#     # 存储结果的列表
#     results = []
#
#     # 遍历所有官能团，查找匹配项
#     for group_name, group_info in functional_groups.items():
#         smarts_pattern = group_info["smarts"]
#         pattern = Chem.MolFromSmarts(smarts_pattern)
#         matches = mol.GetSubstructMatches(pattern)
#
#         # 如果有匹配项，则记录官能团的公式和索引
#         if matches:
#             for match in matches:
#                 results.append((group_info["formula"], match))
#
#     return results
#
# # 示例SMILES
# smiles = 'CCCCC=O.CCOC(=O)C(Cl)Cl'
# results = find_functional_axis(smiles)
# results_dict=dict(results)
# print(results_dict)
# 输出结果
# results_tuple=tuple(results)
# print(results_tuple)
# # for formula, indices in results:
# #     print(f"Formula: {formula}, Indices: {indices}")
# #     result_ele=result_ele.append(indices)
# #
# print(result_ele)
# 给定的元组列表
# data = [
#     ('X (F, Cl, Br, I)', (12,)),
#     ('X (F, Cl, Br, I)', (13,)),
#     ('R-O-R', (8, 7, 9)),
#     ('R-COO-R', (9, 10, 8, 7))
# ]
# # 提取每个元组的第二部分，并将其合成新的独立元组
# new_tuples = tuple(item[1] for item in data)
# # 输出新的独立元组
# print(new_tuples)  # 输出 ((12,), (13,), (8, 7, 9), (9, 10, 8, 7))






















# from rdkit import Chem
#
# # 官能团集合
# functional_groups = {
#     "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#     "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#     "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#     "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#     "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#     "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
#     "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#     "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#     "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#     "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#     "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#     "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#     "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#     "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#     "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#     "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#     "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#     "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#     "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#     "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#     "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#     "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
#     "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
# }
#
# # 从SMILES中提取官能团及其索引
# def extract_functional_groups(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     group_indices = {}
#
#     # 遍历所有官能团
#     for group_name, group_info in functional_groups.items():
#         smarts_pattern = group_info["smarts"]
#         pattern = Chem.MolFromSmarts(smarts_pattern)
#         matches = mol.GetSubstructMatches(pattern)
#
#         if matches:
#             group_indices[group_name] = [match[0] for match in matches]  # 提取匹配的索引位置
#
#     return group_indices
#
# # 示例SMILES
# smiles = 'FC1=C(Br)C=C(C=O)C=C1.NCC(OC)OC'
# groups = extract_functional_groups(smiles)
# print(groups)





# from rdkit import Chem
# from rdkit import Chem
# def find_cyano_group(smiles):
#     # 将SMILES字符串转换为分子对象
#     mol = Chem.MolFromSmiles(smiles)
#     # 检查是否成功解析分子
#     if mol is None:
#         return "Invalid SMILES string"
#     # 定义氰基（N≡C）的SMARTS模式
#     cyano_smarts = "[N]C#C"  # 匹配N≡C部分
#     cyano_pattern = Chem.MolFromSmarts(cyano_smarts)
#     # 检查是否有氰基（N≡C）
#     if mol.HasSubstructMatch(cyano_pattern):
#         return "Cyano group (N≡C) found"
#     else:
#         return "Cyano group (N≡C) not found"
# # 测试用例
# # smiles = "N#CC1=CC=CS1.OC"
# result = find_cyano_group(smiles)
# # 输出结果
# print(result)

# from rdkit import Chem
#
#
# # Define a function to convert SMILES to its canonical form
# def convert_to_canonical(smiles):
#         # Parse the SMILES string to a molecule object
#         mol = Chem.MolFromSmiles(smiles)
#
#         if mol is None:
#                 print(f"Error parsing SMILES: {smiles}")
#                 return None  # Return None if parsing fails
#
#         # Convert to canonical SMILES (ignores hydrogens and ensures consistency)
#         canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True, explicitHydrogen=False)
#
#         return canonical_smiles
#
#
# # Example SMILES list
# smiles_list = [
#        # Ethanol
#         "CC(=O)O",  # Acetic acid
#         "C1=CC=CC=C1",  # Benzene
#         "CC(C(=O)O)N",  # Alanine
#         "C1CCCC1"  # Cyclohexane
# ]
# # Convert each SMILES string to its canonical form
# canonical_smiles_list = [convert_to_canonical(smiles) for smiles in smiles_list]
# # Output canonical SMILES
# for smiles, canonical_smiles in zip(smiles_list, canonical_smiles_list):
#         print(f"Original SMILES: {smiles}  ->  Canonical SMILES: {canonical_smiles}")


#
# # 创建一个特定的官能团作为 SMARTS
# amine_group = Chem.MolFromSmiles("Cc1c(N=C=O)cccc1N=C=O")  # 胺基
# smarts_str = Chem.MolToSmarts(amine_group)
# print(smarts_str)  # 输出：N

# import pandas as pd
# # 示例 DataFrame
# data = {'Name': ['Alice', 'Bob', 'Charlie'],
#         'Age': [25, 30, 35],
#         'City': ['New York', 'Los Angeles', 'Chicago']}
# df = pd.DataFrame(data)
# # 写入 CSV 文件，使用默认的逗号分隔符
# df.to_csv('output.csv', sep=',', index=False)

# import pandas as pd
# # 读取原始 CSV 文件
# df = pd.read_csv('output.csv', header=None)  # 假设没有列头
# # 假设原始文件每个单元格内的数据是以逗号分隔的字符串
# # 我们需要将这些数据分割并展开到多个列
# df_split = df[0].str.split(',', expand=True)
# # 打印分割后的数据
# print(df_split)
# # 将处理后的 DataFrame 保存为新的 CSV 文件
# df_split.to_csv('output1.csv', index=False, header=False)





# import matplotlib.pyplot as plt
# import numpy as np
# # 生成示例数据
# data = np.random.randn(1000)  # 正态分布随机数据
# # 创建直方图
# plt.hist(data, bins=30, edgecolor='black')
# # 添加标题和标签
# plt.title('Frequency Histogram')
# plt.xlabel('Data')
# plt.ylabel('Frequency')
# # 显示图形
# plt.show()



# folder_name_mol = os.path.join(parent_dir, mol_name_suf)
# print(folder_name_mol)
# import pandas as pd
# # 假设有多个字典
import pandas as pd
# 示例数据
# def is_subset(A, B):
#     return B.issubset(A)
# # 示例
# A = {1, 2, 3, 4, 5}
# B = {2, 3}
# if is_subset(A, B):
#     print("A 包含 B 的所有元素")
# else:
#     print("A 不包含 B 的所有元素")
#

#
# import pandas as pd
# # 示例 DataFrame
# df = pd.DataFrame({
#     'col1': ['1', '2', '3', '4.5', 'not_a_number']
# })
# # 将 'col1' 列转换为数字，如果无法转换则变为 NaN
# df['col1'] = pd.to_numeric(df['col1'], errors='coerce')
# print(df)
#




# import pandas as pd
# # 示例 DataFrame
# df = pd.DataFrame({
#     'col1': ['1', '2', '3', '4.5']
# })
# # 将 'col1' 列转换为浮点数
# df['col1'] = df['col1'].astype(float)  # 或者 int，如果需要整数
# print(df)









# data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
# df = pd.DataFrame(data)
# # 加入索引列
# df_with_index = df.reset_index()
# print(df_with_index)





# import pandas as pd
#
# # 示例数组
# arr = [10, 15, 18, 22]
# # 创建一个两列的DataFrame
# data = {'A': [[10, 15, 18, 21], [10, 16, 18, 22], [9, 15, 18, 22], [10, 15, 18, 22]],
#         'B': [100, 101, 102, 103]}  # A列是列表，B列是序号
# df = pd.DataFrame(data)
# # 创建新列用于存储标志符，默认为0
# df['Flag'] = 0
#
#
# # 定义比较函数，检查是否完全相同或仅一个元素不同
# def is_similar(arr1, arr2):
#     # 如果长度不同，直接返回False
#     if len(arr1) != len(arr2):
#         return False
#
#     # 计算两数组中不同元素的数量
#     diff_count = sum([1 for x, y in zip(arr1, arr2) if x != y])
#
#     # 如果不同元素的数量为0或1，返回True
#     return diff_count <= 1
#
# # 遍历DataFrame的每一行，与数组进行比较
# for index, row in df.iterrows():
#     # 如果A列中的列表与数组相同或仅一个元素不同
#     if is_similar(row['A'], arr):
#         df.at[index, 'Flag'] = 1  # 将标志符置为1
# # 将满足条件的行及其序号写入Excel文件
# df[df['Flag'] == 1].to_excel('result.xlsx', index=True)
# print("完成！结果已写入 result.xlsx 文件。")






# import pandas as pd
# # 创建示例数组和DataFrame
# arr = [10, 15, 18, 22]
# # 创建一个两列的DataFrame
# data = {'A': [9, 15, 20, 24], 'B': [100, 101, 102, 103]}  # A列和B列
# df = pd.DataFrame(data)
# # 创建新列用于存储标志符，默认为0
# df['Flag'] = 0
# # 定义一个函数用于比较数组中的每个元素和DataFrame的A列
# def compare_and_flag(array, df):
#     # 遍历数组中的每个元素
#     for value in array:
#         # 遍历DataFrame中的每一行，比较A列
#         for index, row in df.iterrows():
#             # 如果A列的值与数组元素相同或相差1
#             if abs(row['A'] - value) <= 1:
#                 df.at[index, 'Flag'] = 1  # 将标志符置为1
#     return df
# # 调用比较函数
# df = compare_and_flag(arr, df)
# # 将满足条件的行写入Excel文件，包括序号
# df[df['Flag'] == 1].to_excel('result.xlsx', index=True)
# print("完成！结果已写入 result.xlsx 文件。")




# data1 = {
#     "Column1": [1, 2, 3],
#     "Column2": ['A', 'B', 'C']
# }
# data2 = {
#     "Column1": [4, 5, 6],
#     "Column2": ['D', 'E', 'F']
# }
# data3 = {
#     "Column1": [7, 8, 9],
#     "Column2": ['G', 'H', 'I']
# }
# # 创建一个空的 DataFrame 来存储结果
# df_combined = pd.DataFrame()
# # 使用 pd.concat() 代替 append()
# df_combined = pd.concat([df_combined, pd.DataFrame(data1)], ignore_index=True)
# df_combined = pd.concat([df_combined, pd.DataFrame(data2)], ignore_index=True)
# df_combined = pd.concat([df_combined, pd.DataFrame(data3)], ignore_index=True)
# # 将结果保存到 Excel 文件
# df_combined.to_excel("combined_dicts_concat_output.xlsx", index=False)



# import pandas as pd
# # 假设有多个字典
# data1 = {
#     "Column1": [1, 2, 3],
#     "Column2": ['A', 'B', 'C']
# }
# data2 = {
#     "Column1": [4, 5, 6],
#     "Column2": ['D', 'E', 'F']
# }
# data3 = {
#     "Column1": [7, 8, 9],
#     "Column2": ['G', 'H', 'I']
# }
# # 创建一个空的 DataFrame 来存储结果
# df_combined = pd.DataFrame()
# # 依次将字典转换为 DataFrame 并使用 append() 添加
# df_combined = df_combined.append(pd.DataFrame(data1), ignore_index=True)
# df_combined = df_combined.append(pd.DataFrame(data2), ignore_index=True)
# df_combined = df_combined.append(pd.DataFrame(data3), ignore_index=True)
# # 将结果保存到 Excel 文件
# df_combined.to_excel("combined_dicts_append_output.xlsx", index=False)





# import pandas as pd
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# # 读取 Excel 文件中的某一列数据
# df = pd.read_excel('frag_smiles_main_deal_frequency.xlsx', sheet_name='Sheet1')  # 替换为你的 Excel 文件名和表单名
# text_column = df['frag']  # 替换为你的目标列名
# # 将列中的所有文本连接成一个大的字符串
# text = " ".join(text_column.astype(str).tolist())
# # 生成词云
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# # 显示词云
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
# # 保存词云图像（可选）
# wordcloud.to_file("excel_column_wordcloud.png")
#
#
#
#
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# # 输入的一句话
# sentence = "The quick brown fox jumps over the lazy dog."
# # 生成词云
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentence)
# # 显示词云
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
#
#
#
# # 打开TXT文件以写入
# with open('multiple_dicts.txt', 'w') as file:
#     for i, d in enumerate(dict_list):
#         # 写入字典标题
#         file.write(f'Dictionary {i+1}:\n')
#         # 写入字典内容
#         for key, value in d.items():
#             file.write(f'  {key}: {value}\n')
#         # 添加换行符以分隔字典
#         file.write('\n')
#
# print("字典已成功写入TXT文件")


# current_dir = os.getcwd()
# print(f"当前目录: {current_dir}")
# # 倒退到上一个文件夹地址
# parent_dir = os.path.dirname(current_dir)
# print(f"上一级目录: {parent_dir}")
# # 进入上一个文件夹
# os.chdir(parent_dir)
# new_current_dir = os.getcwd()
# print(f"新的当前目录: {new_current_dir}")











# from openpyxl import load_workbook
# # 指定已有的Excel文件路径
# excel_path = 'example.xlsx'
# # 加载已有的Excel工作簿
# wb = load_workbook(filename=excel_path)
# # 选择要写入数据的工作表，这里假设是第一个工作表
# sheet = wb.active
# # 假设有一个列表，包含要写入的多行数据
# data_to_append = [
#     ["姓名", "年龄", "性别"],
#     ["赵六", 25, "男"],
#     ["孙七", 35, "女"]
# ]
# # 确定写入数据的起始行号，这里假设在现有数据之后写入
# start_row = sheet.max_row + 1
# # 逐行写入数据
# for row_data in data_to_append:
#     sheet.append(row_data)
# # 保存Excel文件，覆盖原有文件
# wb.save(filename=excel_path)
#
#
#
#
#
# from openpyxl import Workbook, load_workbook
# # 示例数据列表
# data = [
#     ["Name", "Age", "City"],
#     ["Alice", 30, "New York"],
#     ["Bob", 25, "Los Angeles"],
#     ["Charlie", 35, "Chicago"]
# ]
# # 创建一个新的Excel工作簿和工作表，或者加载现有的Excel文件
# file_path = 'example.xlsx'
# try:
#     # 尝试加载现有的Excel文件
#     wb = load_workbook(file_path)
#     ws = wb.active
# except FileNotFoundError:
#     # 如果文件不存在，则创建一个新的工作簿和工作表
#     wb = Workbook()
#     ws = wb.active
#     ws.title = "Sheet1"
# # 将数据逐行写入工作表
# for row in data:
#     ws.append(row)
# # 保存Excel文件
# wb.save(file_path)
# print(f'已将数据保存到{file_path}')



# from rdkit import Chem
# from rdkit.Chem import Descriptors, rdMolDescriptors
# import pandas as pd
# # 定义你想检测的官能团SMARTS模式
# functional_groups = {
#     'Hydroxyl': '[OX2H]',
#     'Amine': '[NX3;H2,H1;!$(NC=O)]',
#     'Carboxyl': 'C(=O)[OH]',
#     'Aldehyde': '[CX3H1](=O)[#6]',
#     'Ketone': '[CX3](=O)[#6]',
#     'Ester': '[CX3](=O)[OX2H0][#6]',
#     'Ether': '[OD2]([#6])[#6]',
# }
# # 示例分子列表
# smiles_list = [
#     'CCO',  # 乙醇
#     'CC(=O)O',  # 醋酸
#     'CC(=O)C',  # 丙酮
#     'CC(=O)OC',  # 甲酸乙酯
# ]
# # 创建一个空的DataFrame
# columns = ['SMILES'] + list(functional_groups.keys())
# df = pd.DataFrame(columns=columns)
# # 遍历每个分子并检测官能团
# for smiles in smiles_list:
#     mol = Chem.MolFromSmiles(smiles)
#     row = {'SMILES': smiles}
#     for group_name, smarts in functional_groups.items():
#         patt = Chem.MolFromSmarts(smarts)
#         matches = mol.GetSubstructMatches(patt)
#         row[group_name] = len(matches)
#     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
# # 将DataFrame保存为Excel文件d
# df.to_excel('functional_groups.xlsx', index=False)

#
# from rdkit import Chem
# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem import Draw
# import pandas as pd
# # 定义要检测的官能团（示例中的一些常见官能团的 SMARTS 模式）
# functional_groups = {
#     'hydroxyl': '[OX2H]',  # 羟基
#     'carbonyl': '[CX3]=[OX1]',  # 羰基
#     'amine': '[NX3;H2,H1;!$(NC=O)]',  # 胺基
#     'carboxyl': '[CX3](=O)[OX2H1]',  # 羧基
# }
# # 读取分子（这里使用一些示例的 SMILES）
# smiles_list = ["CCO", "CC(=O)O", "CCN", "CC(C(=O)O)O"]
# molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]


# # 创建数据框来保存结果
# data = []
# # 检测每个分子中的官能团
# for mol in molecules:
#     mol_data = {'smiles': Chem.MolToSmiles(mol)}
#     for group_name, smarts in functional_groups.items():
#         pattern = Chem.MolFromSmarts(smarts)
#         mol_data[group_name] = mol.HasSubstructMatch(pattern)
#     data.append(mol_data)
# # 创建 pandas 数据框
# df = pd.DataFrame(data)
# # 保存到 Excel 文件
# df.to_excel("functional_groups.xlsx", index=False)
# print("官能团检测结果已保存到 functional_groups.xlsx")


# def split_string_by_dot(input_string):
#     # 使用split方法按照'.'分割字符串
#     elements = input_string.split('.')
#     return elements
#
# # 示例输入
# input_string = "H2O.CC.O2"
# elements = split_string_by_dot(input_string)
# print("Split Elements:", elements)

# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     if mol is None:
#         raise ValueError("Invalid SMILES string")
#
#     # 标记需要保留的原子
#     atoms_to_keep = set(atom_indices)
#
#     # 创建一个新的分子用于存放提取的部分
#     new_mol = Chem.RWMol()
#     atom_map = {}
#
#     for atom in mol.GetAtoms():
#         if atom.GetIdx() in atoms_to_keep:
#             new_atom_idx = new_mol.AddAtom(atom)
#             atom_map[atom.GetIdx()] = new_atom_idx
#
#     # 复制保留原子之间的键
#     for bond in mol.GetBonds():
#         begin_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()
#         if begin_idx in atoms_to_keep and end_idx in atoms_to_keep:
#             new_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
#
#     # 将新分子转为Mol对象，去芳香化避免错误
#     new_mol = new_mol.GetMol()
#     Chem.Kekulize(new_mol, clearAromaticFlags=True)
#
#     # 获取独立的片段
#     frags = rdmolops.GetMolFrags(new_mol, asMols=True)
#
#     # 获取每个片段的SMILES表示
#     frag_smiles_list = [Chem.MolToSmiles(frag) for frag in frags]
#
#     return frag_smiles_list
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 2, 7, 8, 9]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragments = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragments SMILES:", extracted_fragments)

# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     if mol is None:
#         raise ValueError("Invalid SMILES string")
#
#     # 标记需要保留的原子
#     atoms_to_keep = set(atom_indices)
#
#     # 创建一个新的分子用于存放提取的部分
#     new_mol = Chem.RWMol()
#     atom_map = {}
#
#     for atom in mol.GetAtoms():
#         if atom.GetIdx() in atoms_to_keep:
#             new_atom_idx = new_mol.AddAtom(atom)
#             atom_map[atom.GetIdx()] = new_atom_idx
#
#     # 复制保留原子之间的键
#     for bond in mol.GetBonds():
#         begin_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()
#         if begin_idx in atoms_to_keep and end_idx in atoms_to_keep:
#             new_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
#
#     # 将新分子转为Mol对象，避免芳香化问题
#     new_mol = new_mol.GetMol()
#     Chem.SanitizeMol(new_mol)
#
#     # 获取独立的片段
#     frags = rdmolops.GetMolFrags(new_mol, asMols=True)
#
#     # 获取每个片段的SMILES表示
#     frag_smiles_list = [Chem.MolToSmiles(frag) for frag in frags]
#
#     return frag_smiles_list
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 2, 7, 8, 9]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragments = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragments SMILES:", extracted_fragments)

# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 创建一个新的分子编辑对象
#     rw_mol = Chem.RWMol(mol)
#
#     # 标记需要保留的原子
#     atoms_to_keep = set(atom_indices)
#
#     # 创建一个新的分子用于存放提取的部分
#     new_mol = Chem.RWMol()
#     atom_map = {}
#
#     for atom in rw_mol.GetAtoms():
#         if atom.GetIdx() in atoms_to_keep:
#             new_atom_idx = new_mol.AddAtom(atom)
#             atom_map[atom.GetIdx()] = new_atom_idx
#
#     # 复制保留原子之间的键
#     for bond in rw_mol.GetBonds():
#         begin_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()
#         if begin_idx in atoms_to_keep and end_idx in atoms_to_keep:
#             new_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
#
#     # 获取独立的片段
#     frags = rdmolops.GetMolFrags(new_mol, asMols=True)
#
#     # 获取每个片段的SMILES表示
#     frag_smiles_list = [Chem.MolToSmiles(frag) for frag in frags]
#
#     return frag_smiles_list
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 2, 7, 8, 9]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragments = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragments SMILES:", extracted_fragments)

# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 创建一个新的分子编辑对象
#     rw_mol = Chem.RWMol(mol)
#
#     # 标记需要删除的原子
#     atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetIdx() not in atom_indices]
#
#     # 逆序删除原子，以防索引改变
#     for atom_idx in sorted(atoms_to_remove, reverse=True):
#         rw_mol.RemoveAtom(atom_idx)
#
#     # 将分子转回普通分子对象
#     frag_mol = rw_mol.GetMol()
#
#     # 获取片段的SMILES表示
#     frag_smiles = Chem.MolToSmiles(frag_mol)
#
#     return frag_smiles
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 2, 7, 8, 9]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragment_smiles = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragment SMILES:", extracted_fragment_smiles)

# def is_single_atom(smiles):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#     # 如果解析失败，则返回False
#     if mol is None:
#         return False
#     # 返回分子中原子的数量是否为1
#     return mol.GetNumAtoms() == 1
#
# # 示例输入
# smiles_list = [
#     "C",      # 单个碳原子
#     "CC",     # 乙烷
#     "O",      # 单个氧原子
#     "H2O",    # 水（无效SMILES）
#     "N",      # 单个氮原子
#     "CCO"     # 乙醇
# ]
#
# # 检查每个SMILES字符串
# for smiles in smiles_list:
#     print(f"SMILES: {smiles} -> Is single atom: {is_single_atom(smiles)}")
















# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 创建一个新的分子编辑对象
#     rw_mol = Chem.RWMol(mol)
#
#     # 标记需要删除的原子
#     atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetIdx() not in atom_indices]
#
#     # 逆序删除原子，以防索引改变
#     for atom_idx in sorted(atoms_to_remove, reverse=True):
#         rw_mol.RemoveAtom(atom_idx)
#
#     # 将分子转回普通分子对象
#     frag_mol = rw_mol.GetMol()
#
#     # 获取片段的SMILES表示
#     frag_smiles = Chem.MolToSmiles(frag_mol)
#
#     return frag_smiles

#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0,1,3]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragment_smiles = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragment SMILES:", extracted_fragment_smiles)
#
# #
# def extract_specified_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 创建一个新的分子编辑对象
#     rw_mol = Chem.RWMol(mol)
#
#     # 标记需要保留的原子
#     atoms_to_keep = set(atom_indices)
#
#     # 删除不在原子索引列表中的原子
#     for atom in reversed(rw_mol.GetAtoms()):
#         if atom.GetIdx() not in atoms_to_keep:
#             rw_mol.RemoveAtom(atom.GetIdx())
#
#     # 将分子转回普通分子对象
#     frag_mol = rw_mol.GetMol()
#
#     # 获取片段的SMILES表示
#     frag_smiles = Chem.MolToSmiles(frag_mol)
#
#     return frag_smiles
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 2, 7, 8, 9]  # 指定要保留的原子位置
#
# # 提取片段并输出结果
# extracted_fragment_smiles = extract_specified_atoms(smiles, atom_indices)
# print("Extracted Fragment SMILES:", extracted_fragment_smiles)


# def split_molecule_by_atoms(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 转换为分子编辑对象
#     rw_mol = Chem.RWMol(mol)
#
#     # 切割指定原子位置
#     bonds_to_cut = []
#     for i in atom_indices:
#         atom = rw_mol.GetAtomWithIdx(i)
#         for bond in atom.GetBonds():
#             bonds_to_cut.append(bond.GetIdx())
#
#     # 切割分子
#     fragments = rdmolops.FragmentOnBonds(rw_mol, bonds_to_cut, addDummies=False)
#
#     # 获取片段的SMILES表示
#     frags = Chem.GetMolFrags(fragments, asMols=True)
#     frags_smiles = [Chem.MolToSmiles(frag) for frag in frags]
#
#     return frags_smiles
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [1, 6]  # 指定要切割的原子位置
#
# # 切割分子并输出结果
# fragments_smiles = split_molecule_by_atoms(smiles, atom_indices)
# for i, frag_smiles in enumerate(fragments_smiles):
#     print(f"Fragment {i + 1}: {frag_smiles}")
#
#
# def split_molecule(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 获取要切割的键
#     bonds_to_cut = []
#     for i in atom_indices:
#         atom = mol.GetAtomWithIdx(i)
#         for bond in atom.GetBonds():
#             bond_idx = bond.GetIdx()
#             if bond_idx not in bonds_to_cut:
#                 bonds_to_cut.append(bond_idx)
#
#     # 切割分子
#     fragments = FragmentOnBonds(mol, bonds_to_cut)
#
#     # 获取片段的SMILES表示
#     frags_smiles = Chem.MolToSmiles(fragments)
#
#     return frags_smiles
#
#
# # 示例输入：SMILES字符串和原子位置
# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
# atom_indices = [0, 1, 10]  # 指定要切割的原子位置
#
# # 切割分子并输出结果
# fragments_smiles = split_molecule(smiles, atom_indices)
# print("Fragmented SMILES:", fragments_smiles)

#
# # 定义常见官能团的SMARTS模式及其化学式
# functional_groups = {
#     "Alkane": {"smarts": "[CX4]", "formula": "C-H"},
#     "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#     "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#     "Aromatic": {"smarts": "c1ccccc1", "formula": "C6H5"},
#     "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#     "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#     "Phenol": {"smarts": "c1ccc(O)cc1", "formula": "C6H5OH"},
#     "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#     "Aldehyde": {"smarts": "[CX3H1](=O)[#6]", "formula": "R-CHO"},
#     "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
#     "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#     "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#     "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#     "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#     "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#     "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#     "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#     "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#     "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#     "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#     "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#     "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#     "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#     "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#     "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#     "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"}
# }
#
#
# def identify_functional_groups(smiles):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 存储检测到的官能团及其化学式
#     detected_groups = []
#
#     # 查找官能团
#     for group_name, properties in functional_groups.items():
#         pattern = Chem.MolFromSmarts(properties["smarts"])
#         if mol.HasSubstructMatch(pattern):
#             detected_groups.append((group_name, properties["formula"]))
#
#     return detected_groups
#
#
# # 示例输入：SMILES字符串
# smiles = "CC(=O)O"  # 乙酸
#
# # 识别并输出官能团
# detected_groups = identify_functional_groups(smiles)
# for group_name, formula in detected_groups.items():
#     print(f"Detected functional group: {group_name}, Chemical formula: {formula}")

# # 定义常见官能团的SMARTS模式
# functional_groups = {
#     "Alkane": "[CX4]",
#     "Alkene": "[CX3]=[CX3]",
#     "Alkyne": "[CX2]#[CX2]",
#     "Aromatic": "c1ccccc1",
#     "Halide": "[F,Cl,Br,I]",
#     "Alcohol": "[OX2H]",
#     "Phenol": "c1ccc(O)cc1",
#     "Ether": "[OD2]([#6])[#6]",
#     "Aldehyde": "[CX3H1](=O)[#6]",
#     "Ketone": "[CX3](=O)[#6]",
#     "Carboxylic Acid": "[CX3](=O)[OX2H1]",
#     "Ester": "[CX3](=O)[OX2][#6]",
#     "Amide": "[NX3][CX3](=[OX1])[#6]",
#     "Amine": "[NX3][#6]",
#     "Nitrate": "[NX3](=O)([OX1-])[OX1-]",
#     "Nitro": "[NX3](=O)[OX1-]",
#     "Sulfonic Acid": "S(=O)(=O)[O-]",
#     "Thiol": "[SX2H]",
#     "Thioether": "[SX2][#6]",
#     "Disulfide": "[SX2][SX2]",
#     "Sulfoxide": "[SX3](=O)[#6]",
#     "Sulfone": "[SX4](=O)(=O)[#6]",
#     "Phosphine": "[PX3]",
#     "Phosphate": "P(=O)(O)(O)O",
#     "Isocyanate": "N=C=O",
#     "Isothiocyanate": "N=C=S"
# }
#
#
# def identify_functional_groups(smiles):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 存储检测到的官能团
#     detected_groups = set()
#
#     # 查找官能团
#     for group_name, smarts in functional_groups.items():
#         pattern = Chem.MolFromSmarts(smarts)
#         if mol.HasSubstructMatch(pattern):
#             detected_groups.add(group_name)
#
#     return detected_groups
#
#
# # 示例输入：SMILES字符串
# #smiles = "CC(=O)O"  # 乙酸
# smiles="NC(=S)Nc1ccccc1"
# # 识别并输出官能团
# detected_groups = identify_functional_groups(smiles)
# print("Detected functional groups:", detected_groups)

## function group chinese
# Alkane: "[CX4]" 匹配所有四价的碳原子，即饱和碳（烷烃）。
# Alkene: "[CX3]=[CX3]" 匹配双键碳原子（烯烃）。
# Alkyne: "[CX2]#[CX2]" 匹配三键碳原子（炔烃）。
# Aromatic: "c1ccccc1" 匹配苯环。
# Halide: "[F,Cl,Br,I]" 匹配卤素原子（氟、氯、溴、碘）。
# Alcohol: "[OX2H]" 匹配羟基（酒精）。
# Phenol: "c1ccc(O)cc1" 匹配苯酚。
# Ether: "OD2[#6]" 匹配醚。
# Aldehyde: "CX3H1[#6]" 匹配醛。
# Ketone: "CX3[#6]" 匹配酮。
# Carboxylic Acid: "CX3[OX2H1]" 匹配羧酸。
# Ester: "CX3[OX2][#6]" 匹配酯。
# Amide: "[NX3]CX3[#6]" 匹配酰胺。
# Amine: "[NX3][#6]" 匹配胺。
# Nitrate: "NX3([OX1-])[OX1-]" 匹配硝酸盐。
# Nitro: "NX3[OX1-]" 匹配硝基。
# Sulfonic Acid: "S(=O)(=O)[O-]" 匹配磺酸。
# Thiol: "[SX2H]" 匹配硫醇。
# Thioether: "[SX2][#6]" 匹配硫醚。
# Disulfide: "[SX2][SX2]" 匹配二硫化物。
# Sulfoxide: "SX3[#6]" 匹配亚砜。
# Sulfone: "SX4(=O)[#6]" 匹配砜。
# Phosphine: "[PX3]" 匹配膦。
# Phosphate: "P(=O)(O)(O)O" 匹配磷酸。
# Isocyanate: "N=C=O" 匹配异氰酸酯。
# Isothiocyanate: "N=C=S" 匹配异硫氰酸酯。




#
# # 定义常见官能团的SMARTS模式
# functional_groups = {
#     "Hydroxyl": "[O]",
#     "Carbonyl": "[C=O]",
#     "Carboxyl": "[C(=O)O]",
#     "Amino": "[NH2]",
#     "Phenyl": "c1ccccc1",
#     "Ether": "[C-O-C]",
#     "Ester": "[C(=O)O-C]",
#     "Amide": "[C(=O)N]",
#     "Nitrate": "[N+](=O)[O-]"
# }
#
#
# def identify_functional_groups(smiles):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 存储检测到的官能团
#     detected_groups = set()
#
#     # 查找官能团
#     for group_name, smarts in functional_groups.items():
#         pattern = Chem.MolFromSmarts(smarts)
#         if mol.HasSubstructMatch(pattern):
#             detected_groups.add(group_name)
#
#     return detected_groups
#
#
# # 示例输入：SMILES字符串
# smiles = "CC(=O)O"  # 乙酸
#
# # 识别并输出官能团
# detected_groups = identify_functional_groups(smiles)
# print("Detected functional groups:", detected_groups)
#
# # from rdkit import Chem
# #
# # # 定义常见官能团的SMARTS模式
# # functional_groups = {
# #     "Hydroxyl": "[OH]",
# #     "Carbonyl": "[C=O]",
# #     "Carboxyl": "[C(=O)O]",
# #     "Amino": "[NH2]",
# #     "Phenyl": "c1ccccc1",
# #     "Ether": "[C-O-C]",
# #     "Ester": "[C(=O)O-C]",
# #     "Amide": "[C(=O)N]",
# #     "Nitrate": "[N+](=O)[O-]"
# # }
# #
# #
# # def check_functional_groups(smiles, atom_indices):
# #     # 使用RDKit解析SMILES字符串
# #     mol = Chem.MolFromSmiles(smiles)
# #
#     # 存储每个原子是否属于某个官能团
#     atom_functional_groups = {idx: [] for idx in atom_indices}
#
#     # 查找官能团
#     for group_name, smarts in functional_groups.items():
#         pattern = Chem.MolFromSmarts(smarts)
#         matches = mol.GetSubstructMatches(pattern)
#         for match in matches:
#             for idx in match:
#                 if idx in atom_indices:
#                     atom_functional_groups[idx].append(group_name)
#
#     return atom_functional_groups
#
#
# # 示例输入：SMILES字符串和对应的原子序号列表
# smiles = "CC(=O)O"  # 乙酸
# atom_indices = [0, 1]  # 原子序号列表
#
# # 检查这些原子是否属于某个官能团
# result = check_functional_groups(smiles, atom_indices)
#
# # 输出结果
# for atom_idx, groups in result.items():
#     print(f"Atom {atom_idx}: {groups}")




# from rdkit import Chem
#
# # 定义官能团SMARTS模式
# functional_groups = {
#     "Hydroxyl": "[OH]",
#     "Carbonyl": "[C=O]",
#     "Carboxyl": "[C(=O)O]",
#     "Amino": "[NH2]",
#     "Phenyl": "c1ccccc1",
#     "Ether": "[C-O-C]",
#     "Ester": "[C(=O)O-C]",
#     "Amide": "[C(=O)N]",
#     "Nitrate": "[N+](=O)[O-]"
# }
#
#
# def check_functional_groups(smiles, atom_indices):
#     # 使用RDKit解析SMILES字符串
#     mol = Chem.MolFromSmiles(smiles)
#
#     # 存储每个原子是否属于某个官能团
#     atom_functional_groups = {idx: [] for idx in atom_indices}
#
#     # 查找官能团
#     for group_name, smarts in functional_groups.items():
#         pattern = Chem.MolFromSmarts(smarts)
#         matches = mol.GetSubstructMatches(pattern)
#         for match in matches:
#             for idx in match:
#                 if idx in atom_indices:
#                     atom_functional_groups[idx].append(group_name)
#
#     return atom_functional_groups
#
#
# # 输入SMILES字符串和对应的原子序号列表
# smiles = "CC(=O)O"
# atom_indices = [1, 2, 3, 4]  # 例子中的原子序号
#
# # 检查这些原子是否属于某个官能团
# result = check_functional_groups(smiles, atom_indices)
#
# # 输出结果
# for atom_idx, groups in result.items():
#     print(f"Atom {atom_idx}: {groups}")

#
# from rdkit import Chem
# from rdkit.Chem import rdMolDescriptors
#
# # 输入SMILES字符串和对应的原子序号列表
# smiles = "CCO"
# atom_indices = [1, 2, 3]  # 举例的原子序号
#
# # 使用RDKit解析SMILES字符串
# mol = Chem.MolFromSmiles(smiles)
#
# # 定义常见官能团SMARTS模式
# functional_groups = {
#     "Hydroxyl": "[OH]",
#     "Carbonyl": "[C=O]",
#     "Carboxyl": "[C(=O)O]",
#     "Amino": "[NH2]",
#     "Phenyl": "c1ccccc1",
#     "Ether": "[C-O-C]",
#     "Ester": "[C(=O)O-C]",
#     "Amide": "[C(=O)N]",
#     "Nitrate": "[N+](=O)[O-]"
# }
#
# # 查找官能团
# detected_groups = []
# for group_name, smarts in functional_groups.items():
#     pattern = Chem.MolFromSmarts(smarts)
#     if mol.HasSubstructMatch(pattern):
#         detected_groups.append(group_name)
#
# # 输出存在的官能团
# print("Detected functional groups:", detected_groups)

#
# from rdkit import Chem
# from rdkit.Chem import Draw
#
# def find_functional_groups(smiles, functional_groups):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#
#     results = {}
#     for name, smarts in functional_groups.items():
#         patt = Chem.MolFromSmarts(smarts)
#         matches = mol.GetSubstructMatches(patt)
#         if matches:
#             results[name] = matches
#
#     return results
#
# # 定义官能团的SMARTS
# functional_groups = {
#     "Hydroxyl": "[OX2H]",        # 羟基
#     "Carbonyl": "[CX3]=[OX1]",   # 羰基
#     "Amine": "[NX3;H2,H1;!$(NC=O)]", # 胺基
#     "Carboxyl": "C(=O)[OH]",     # 羧基
#     # 可以添加更多官能团
# }
#
# # 测试示例
# smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 乙酸苯酯
#
# results = find_functional_groups(smiles, functional_groups)
#
# if results:
#     for fg, matches in results.items():
#         print(f"{fg} found at positions: {matches}")
# else:
#     print("No functional groups found.")
#
# # 绘制分子结构并标注官能团位置
# mol = Chem.MolFromSmiles(smiles)
# Draw.MolToImage(mol, highlightAtoms=[atom for match in results.values() for group in match for atom in group])
#

# def find_functional_groups(smiles, functional_groups):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#
#     results = {}
#     for name, smarts in functional_groups.items():
#         patt = Chem.MolFromSmarts(smarts)
#         matches = mol.GetSubstructMatches(patt)
#         if matches:
#             results[name] = matches
#
#     return results
#
# # 定义官能团的SMARTS
# functional_groups = {
#     "Hydroxyl": "[OX2H]",        # 羟基
#     "Carbonyl": "[CX3]=[OX1]",   # 羰基
#     "Amine": "[NX3;H2,H1;!$(NC=O)]", # 胺基
#     "Carboxyl": "C(=O)[OH]",     # 羧基
#     # 可以添加更多官能团
# }
#
# # 测试示例
# smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 乙酸苯酯
# results = find_functional_groups(smiles, functional_groups)
#
# if results:
#     for fg, matches in results.items():
#         print(f"{fg} found at positions: {matches}")
# else:
#     print("No functional groups found.")



# def find_and_remove_non_letters(input_string):
#     non_letter_positions = []
#     clean_string = ""
#
#     for i, char in enumerate(input_string):
#         if not char.isalpha():
#             non_letter_positions.append(i)
#         else:
#             clean_string += char
#
#     return non_letter_positions, clean_string
#
# # 测试示例
# input_string = "Hello, World! 123"
# positions, clean_string = find_and_remove_non_letters(input_string)
#
# print("非字母字符的位置:", positions)
# print("删除非字母字符后的字符串:", clean_string)


# import re
#
# def find_non_alphanumeric_positions(s):
#     # 使用正则表达式找到所有非字母和数字的元素位置
#     return [(match.start(), match.group()) for match in re.finditer(r'\W', s)]
#
# # 示例字符串
# s = "Hello, World! 123."
#
# positions = find_non_alphanumeric_positions(s)
# print(positions)

# import re
# import numpy as np
#
# def find_non_alphanumeric_positions(s):
#     # 使用正则表达式找到所有非字母和数字的元素位置
#     return [(match.start(), match.group()) for match in re.finditer(r'\W', s)]
#
# def positions_matrix(positions, length):
#     # 创建一个与字符串长度相同的零矩阵
#     matrix = np.zeros((1, length), dtype=int)
#     for pos, char in positions:
#         matrix[0, pos] = pos + 1  # 位置从1开始编号
#     return matrix
#
# # 示例字符串
# s = "Hello, World! 123."
#
# positions = find_non_alphanumeric_positions(s)
# matrix = positions_matrix(positions, len(s))
#
# print("Positions and characters:", positions)
# print("Positions matrix:\n", matrix)

# import re
#
#
# def find_non_alphanumeric_positions_and_remove(s):
#     # 使用正则表达式找到所有非字母和数字的元素位置
#     positions = [match.start() for match in re.finditer(r'\W', s)]
#
#     # 删除非字母和数字元素后生成新的字符串
#     new_string = re.sub(r'\W', '', s)
#
#     return positions, new_string
#
#
# # 示例字符串
# s = "Hello, World! 123."
#
# positions, new_string = find_non_alphanumeric_positions_and_remove(s)
#
# print("Positions of non-alphanumeric characters:", positions)
# print("String after removing non-alphanumeric characters:", new_string)


# import re
#
#
# def find_non_alphanumeric_positions_and_remove(lst):
#     # 找到所有非字母和数字的元素位置
#     positions = [i for i, elem in enumerate(lst) if not re.match(r'\w', str(elem))]
#
#     # 删除非字母和数字元素后生成新的列表
#     new_list = [elem for i, elem in enumerate(lst) if i not in positions]
#
#     return positions, new_list
#
#
# # 示例列表
# lst = ['H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!', ' ', '1', '2', '3', '.']
#
# positions, new_list = find_non_alphanumeric_positions_and_remove(lst)
#
# print("Positions of non-alphanumeric characters:", positions)
# print("List after removing non-alphanumeric characters:", new_list)



