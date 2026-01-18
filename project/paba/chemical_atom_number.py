from rdkit import Chem


# 查找并返回指定元素的原子编号
def get_atom_indices_by_symbol(molecule_smiles, element_symbol):
    # 将SMILES字符串转化为分子对象
    mol = Chem.MolFromSmiles(molecule_smiles)
    if not mol:
        return "Invalid SMILES"
    # 查找所有指定元素的原子
    atom_indices = []
    # 遍历所有原子，查找与指定元素符号匹配的原子
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element_symbol:  # 检查是否是指定元素
            atom_indices.append(atom.GetIdx())  # 获取原子的索引

    return atom_indices

if __name__ == "__main__":
    # 示例分子
    molecule_smiles = "CC1=CC=C(C=O)C=C1.C2CCCCN2"
    # 输入你想要查找的元素符号
    element_symbol = 'N'  # 可以替换为 'C', 'N', 'H', 等等
    atom_indices = get_atom_indices_by_symbol(molecule_smiles, element_symbol)
    # 输出指定元素的原子编号
    if atom_indices:
        print(f"The indices of {element_symbol} atoms in the molecule are: {atom_indices}")
    else:
        print(f"No {element_symbol} atoms found.")