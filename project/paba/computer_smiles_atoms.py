from rdkit import Chem
import os
import numpy as np
import pandas as pd
# 定义官能团的 SMARTS 表达式
functional_groups = {
    "Alkane": {"smarts": "[CX4]"},
    "Alkene": {"smarts": "[CX3]=[CX3]"},
    "Alkyne": {"smarts": "[CX2]#[CX2]"},
    "Aromatic": {"smarts": "c1ccccc1"},
    "Halide": {"smarts": "[F,Cl,Br,I]"},
    "Alcohol": {"smarts": "[OX2H]"},
    "Phenol": {"smarts": "c1ccc(O)cc1"},
    "Ether": {"smarts": "[OD2]([#6])[#6]"},
    "Aldehyde": {"smarts": "[CX3H1](=O)[#6]"},
    "Ketone": {"smarts": "[CX3](=O)[#6]"},
    "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]"},
    "Ester": {"smarts": "[CX3](=O)[OX2][#6]"},
    "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]"},
    "Amine": {"smarts": "[NX3][#6]"},
    "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]"},
    "Nitro": {"smarts": "[NX3](=O)[OX1-]"},
    "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]"},
    "Thiol": {"smarts": "[SX2H]"},
    "Thioether": {"smarts": "[SX2][#6]"},
    "Disulfide": {"smarts": "[SX2][SX2]"},
    "Sulfoxide": {"smarts": "[SX3](=O)[#6]"},
    "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]"},
    "Phosphine": {"smarts": "[PX3]"},
    "Phosphate": {"smarts": "P(=O)(O)(O)O"},
    "Isocyanate": {"smarts": "N=C=O"},
    "Isothiocyanate": {"smarts": "N=C=S"}
}

def analyze_smiles(smiles):
    # 解析 SMILES 表达式并生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("无效的 SMILES 表达式")
        return None, None

    # 统计原子总个数
    atom_count = sum(1 for atom in mol.GetAtoms())

    # 统计官能团出现次数，仅显示找到的官能团
    found_functional_groups = {}
    for group_name, props in functional_groups.items():
        smarts = props["smarts"]
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) > 0:
                found_functional_groups[group_name] = len(matches)

    return atom_count, found_functional_groups


# 示例
smiles = "CC(=O)OCC"  # 乙酸乙酯
atom_count, found_functional_groups = analyze_smiles(smiles)

print("SMILES 表达式中的总原子个数:", atom_count)
print("找到的官能团及其个数:", found_functional_groups)

current_dir = os.getcwd()
current_dir_next=os.path.join(current_dir,'data')

df_reactive = pd.read_excel((os.path.join(current_dir_next, '0911mark_flag_human_sprit_to_x_1020_p_gold_(4,4) _v01.xlsx')))
#df_smaple= pd.read_excel((os.path.join(input_files_smiles, 'frag_location_main_0.98.xlsx')))
df_reactive_example=df_reactive[['SMILES','atoms','functiona_group']]

for index_criterion, row_criterion in df_reactive_example.iterrows():
    value_smiles = row_criterion['SMILES']
    atom_count, found_functional_groups=analyze_smiles(value_smiles)
    df_reactive_example.at[index_criterion, 'atoms'] = atom_count
    #df_reactive_example.at[index_criterion, 'functional_group'] = found_functional_groups
