import os
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_dir))
label_file_name = f'data/input_smiles/uspto_50k/uspto_yuming_50k_localtransform'
input_folder=os.path.join(parent_dir,label_file_name)
excel_detail='USPTO_50K_yuming_valid.xlsx'
excel_path=os.path.join(input_folder,excel_detail)
txt_path='USPTO_50K_yuming_valid_combine.txt'
output_txt_path=os.path.join(input_folder,txt_path)

# # 1. Excel 文件路径（修改为你自己的路径）
# excel_path = 'USPTO_50K_yuming_valid.xlsx'  # 例如：'data.xlsx'
#
# # 2. 读取数据
df = pd.read_excel(excel_path)
#
# # 3. 输出 txt 文件路径
# output_txt_path = ('USPTO_50K_yuming_valid_combine.txt')

# 4. 打开文件写入所有行（每行一个 reaction,reactant）
with open(output_txt_path, 'w', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        reaction = str(row['local_atom_map']).strip()
        reactant = str(row['reactant']).strip()
        f.write(f"{reaction},{reactant}\n")

print(f"✅ 所有对数据已保存到：{output_txt_path}")