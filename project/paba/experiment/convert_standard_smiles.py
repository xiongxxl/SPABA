import pandas as pd
from rdkit import Chem
import os

def standardize_single_smiles(smiles):
    """
    将单个 SMILES 转换为标准化 SMILES，无法解析时返回原始 SMILES。
    """
    try:
        # 转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 转换为标准化的 SMILES
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            # 无法解析时返回原始 SMILES
            return smiles
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return smiles


def process_smiles_in_excel(input_file, output_file):
    """
    读取 Excel 文件，转换 SMILES 列为标准化 SMILES，并保存到新的 Excel 文件。

    Args:
        input_file (str): 输入的 Excel 文件路径。
        output_file (str): 输出的 Excel 文件路径。
    """
    # 读取 Excel 文件
    df = pd.read_excel(input_file)

    # 假设 SMILES 列的名称为 'SMILES'
    if 'smiles' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'SMILES' 列")

    # 对 SMILES 列进行标准化处理
    df['Standardized_SMILES'] = df['smiles'].apply(standardize_single_smiles)

    # 保存结果到新的 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"处理后的结果已保存到: {output_file}")



current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
# save fragments
statistics_path='data/result/statistics_reactive/double_70'
fragment_excel=f'frag_smiles_main_{ratio_adaptive}.xlsx'  #save functional group path
statistics_fragment_path= os.path.join(statistics_path, fragment_excel)
folder_fragment_statistics=os.path.join(parent_dir, statistics_fragment_path)
# 示例调用
input_file = "double_criterion_70.xlsx"  # 输入的 Excel 文件，包含 'SMILES' 列
output_file = "output_smiles.xlsx"  # 输出的 Excel 文件，保存标准化的 SMILES
process_smiles_in_excel(input_file, output_file)