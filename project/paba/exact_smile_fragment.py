from rdkit import Chem
from rdkit.Chem import rdmolops

def split_and_remove_single_elements(input_string):
    # 使用 split 方法按照 '.' 分割字符串
    elements = input_string.split('.')
    # 过滤掉单个字符的元素
    filtered_elements = [element for element in elements if len(element) > 1]
    return filtered_elements

def split_string_by_dot(input_string):
    # 使用split方法按照'.'分割字符串
    elements = input_string.split('.')
    return elements

def extract_specified_atoms(smiles, atom_indices):
    # 使用RDKit解析SMILES字符串
    mol = Chem.MolFromSmiles(smiles)

    # 创建一个新的分子编辑对象
    rw_mol = Chem.RWMol(mol)

    # 标记需要删除的原子
    atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetIdx() not in atom_indices]

    # 逆序删除原子，以防索引改变
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(atom_idx)

    # 将分子转回普通分子对象
    frag_mol = rw_mol.GetMol()

    # 获取片段的SMILES表示
    frag_smiles = Chem.MolToSmiles(frag_mol)
    keep_single_elements = split_string_by_dot(frag_smiles)
    del_single_elements = split_and_remove_single_elements(frag_smiles)
    return  del_single_elements,keep_single_elements,frag_smiles




if __name__ == "__main__":
    # 示例输入：SMILES字符串和原子位置
    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    # atom_indices = [0,1,3]  # 指定要保留的原子位置
    smiles = "C=CC1=CC=CC=C1.O=CCC"  # 阿司匹林
    atom_indices = [0,5,8,9]  # 指定要保留的原子位置

    # 提取片段并输出结果
    keep_single_elements,del_single_elements,frag_smiles = extract_specified_atoms(smiles, atom_indices)
    print("Extracted Fragment SMILES:", keep_single_elements,del_single_elements,frag_smiles)