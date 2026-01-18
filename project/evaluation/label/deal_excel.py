# import pandas as pd
# import ast
#
# # 读取 Excel 文件
# df = pd.read_excel("reactive_atom_by_template_v03.xlsx")
#
# # 假设这一列叫 "col"
# def merge_tuples(cell):
#     try:
#         # 去掉空格并拆分
#         parts = cell.split('.')
#         # 把字符串转成元组对象
#         t1 = ast.literal_eval(parts[0])
#         t2 = ast.literal_eval(parts[1])
#         # 合并为列表
#         return list(t1 + t2)
#     except Exception as e:
#         return None  # 或者返回 cell 保留原值
#
# # 应用到列
# df["merged"] = df["atom_indices"].apply(merge_tuples)
#
# # 保存结果
# df.to_excel("reactive_atom_by_template_v03_merge.xlsx", index=False)

# ## deal with smiles
# import pandas as pd
# import ast
#
# # 读取 Excel 文件
# df = pd.read_excel("reactive_atom_by_template_v03.xlsx")
#
# def to_list(cell):
#     try:
#         if '.' in cell:
#             # 情况1：带两个元组
#             parts = cell.split('.')
#             t1 = ast.literal_eval(parts[0])
#             t2 = ast.literal_eval(parts[1])
#             return list(t1 + t2)
#         else:
#             # 情况2：只有一个元组
#             t = ast.literal_eval(cell)
#             return list(t)
#     except Exception as e:
#         return None  # 或保留原值 cell
#
# # 应用函数
# df["merged"] = df["atom_indices"].apply(to_list)
#
# # 保存结果
# df.to_excel("reactive_atom_by_template_v03_merge.xlsx", index=False)




# ## make to  canoical smiles
# import pandas as pd
# from rdkit import Chem
#
# def canonicalize_smiles(smiles: str) -> str:
#     """
#     将分子或反应 SMILES 转换为标准 Canonical SMILES（不含显式氢）
#     """
#     if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
#         return ""
#     try:
#         if ">" in smiles:  # 反应 SMILES
#             parts = smiles.split(">")
#             reactants = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in parts[0].split('.') if s
#             ])
#             reagents = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in (parts[1].split('.') if len(parts) > 1 else []) if s
#             ])
#             products = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in (parts[2].split('.') if len(parts) > 2 else []) if s
#             ])
#             return f"{reactants}>{reagents}>{products}"
#         else:  # 单个分子
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return ""
#             return Chem.MolToSmiles(mol, canonical=True)
#     except Exception as e:
#         return ""
#
# # ====== 文件路径设置 ======
# input_file = "data/reactive_atom_by_template_v01.xlsx"  # 输入 Excel 文件路径
# output_file = "data/reactive_atom_by_template_v01_cannonical.xlsx"  # 输出文件路径
# sheet_name = 0                  # 工作表名或索引
# smiles_column = "reactions"        # 输入列名（修改为你的列名）
#
# # ====== 读取 Excel ======
# df = pd.read_excel(input_file, sheet_name=sheet_name)
#
# # ====== 转换并生成新列 ======
# df["Canonical_SMILES"] = df[smiles_column].apply(canonicalize_smiles)
#
# # ====== 写出结果 ======
# df.to_excel(output_file, index=False)
# print(f"✅ 转换完成，结果已保存到 {output_file}")
#
#
# import pandas as pd
# import ast
#
# # 读取 Excel 文件
# df = pd.read_excel("your_file.xlsx")
#
# # 假设这一列叫 "col"
# def merge_tuples(cell):
#     try:
#         # 去掉空格并拆分
#         parts = cell.split('.')
#         # 把字符串转成元组对象
#         t1 = ast.literal_eval(parts[0])
#         t2 = ast.literal_eval(parts[1])
#         # 合并为列表
#         return list(t1 + t2)
#     except Exception as e:
#         return None  # 或者返回 cell 保留原值
#
# # 应用到列
# df["merged"] = df["col"].apply(merge_tuples)
#
# # 保存结果
# df.to_excel("merged_output.xlsx", index=False)


#
# import pandas as pd
# import ast
#
# # 读取 Excel 文件
# df = pd.read_excel("uspto_yang_reactive_atom_orgin_50k_indices.xlsx")
#
# def to_list(cell):
#     try:
#         if '.' in cell:
#             # 情况1：带两个元组
#             parts = cell.split('.')
#             t1 = ast.literal_eval(parts[0])
#             t2 = ast.literal_eval(parts[1])
#             return list(t1 + t2)
#         else:
#             # 情况2：只有一个元组
#             t = ast.literal_eval(cell)
#             return list(t)
#     except Exception as e:
#         return None  # 或保留原值 cell
#
# # 应用函数
# df["reactive_atoms"] = df["reactant_indices"].apply(to_list)
#
# # 保存结果
# df.to_excel("uspto_yang_reactive_atom_orgin_50k_indices_atoms.xlsx", index=False)



# import pandas as pd
# from rdkit import Chem
#
# def canonicalize_smiles(smiles: str) -> str:
#     """
#     将分子或反应 SMILES 转换为标准 Canonical SMILES（不含显式氢）
#     """
#     if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
#         return ""
#     try:
#         if ">" in smiles:  # 反应 SMILES
#             parts = smiles.split(">")
#             reactants = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in parts[0].split('.') if s
#             ])
#             reagents = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in (parts[1].split('.') if len(parts) > 1 else []) if s
#             ])
#             products = '.'.join([
#                 Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
#                 for s in (parts[2].split('.') if len(parts) > 2 else []) if s
#             ])
#             return f"{reactants}>{reagents}>{products}"
#         else:  # 单个分子
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return ""
#             return Chem.MolToSmiles(mol, canonical=True)
#     except Exception as e:
#         return ""
#
# # ====== 文件路径设置 ======
# input_file = "input.xlsx"       # 输入 Excel 文件路径
# output_file = "output.xlsx"     # 输出文件路径
# sheet_name = 0                  # 工作表名或索引
# smiles_column = "SMILES"        # 输入列名（修改为你的列名）
#
# # ====== 读取 Excel ======
# df = pd.read_excel(input_file, sheet_name=sheet_name)
#
# # ====== 转换并生成新列 ======
# df["Canonical_SMILES"] = df[smiles_column].apply(canonicalize_smiles)
#
# # ====== 写出结果 ======
# df.to_excel(output_file, index=False)
# print(f"✅ 转换完成，结果已保存到 {output_file}")



# ## 将smiles分立的编号变成整体编号
# import pandas as pd
# from rdkit import Chem
#
#
# def adjust_atom_indices(reactant: str, atom_indices: str) -> str:
#     """
#     仅处理双分子 reactant 的 atom_indices。
#     """
#     # 如果是单分子，不处理，直接返回原 atom_indices
#     if '.' not in reactant:
#         return atom_indices
#
#     atom_indices = atom_indices.strip("()")
#
#     first_reactant_smiles = reactant.split(".")[0]
#     mol = Chem.MolFromSmiles(first_reactant_smiles)
#     num_atoms = mol.GetNumAtoms()  # 显式原子数
#
#     pre, post = atom_indices.split(".")
#
#     pre_numbers = pre.strip("()").split(",")
#     pre_tuple = "(" + ",".join(pre_numbers) + ")"
#
#     post_numbers = post.strip("()").split(",")
#     post_numbers = [str(int(i) + num_atoms) for i in post_numbers]
#     post_tuple = "(" + ",".join(post_numbers) + ")"
#
#     return pre_tuple + "." + post_tuple
#
#
# # ---------------------------
# # 读取 Excel
# # ---------------------------
# df = pd.read_excel("uspto_yang_reactive_atom_orgin_50k.xlsx")
#
# # 仅处理双分子反应物，单分子保持原样
# df['reactant_indices'] = df.apply(lambda row: adjust_atom_indices(row['reactant'], row['atom_indices']), axis=1)
#
# # 保存到新的 Excel
# df.to_excel("uspto_yang_reactive_atom_orgin_50k_indices.xlsx", index=False)
#
# print("已生成 reactant_indices 并保存到 data_with_indices.xlsx")