from rdkit import Chem
from rdkit.Chem import Draw, AllChem


def extract_reaction_center(reactants_smiles, reaction_smirks):
    """
    从反应物和反应模板中提取反应中心并可视化

    参数:
        reactants_smiles (str): 反应物的SMILES字符串
        reaction_smirks (str): 反应模板的SMIRKS字符串

    返回:
        None (直接显示图像)
    """
    # 1. 准备反应物和反应模板
    reactants = Chem.MolFromSmiles(reactants_smiles)
    rxn = AllChem.ReactionFromSmarts(reaction_smirks)

    # 2. 应用反应模板
    products = rxn.RunReactants((reactants,))
    if not products:
        print("反应物不匹配该反应模板！")
        return

    # 3. 获取原子映射信息
    product = products[0][0]  # 取第一个产物
    atom_maps = {}
    for atom in product.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom_maps[int(atom.GetProp("molAtomMapNumber"))] = atom.GetIdx()

    # 4. 识别反应中心（变化的键）
    reaction_center_bonds = set()
    reactants_bonds = set()
    products_bonds = set()

    # 获取反应物中的键（原子映射对）
    for bond in reactants.GetBonds():
        beg_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if beg_atom.HasProp("molAtomMapNumber") and end_atom.HasProp("molAtomMapNumber"):
            map_beg = int(beg_atom.GetProp("molAtomMapNumber"))
            map_end = int(end_atom.GetProp("molAtomMapNumber"))
            reactants_bonds.add(tuple(sorted((map_beg, map_end))))

    # 获取产物中的键（原子映射对）
    for bond in product.GetBonds():
        beg_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if beg_atom.HasProp("molAtomMapNumber") and end_atom.HasProp("molAtomMapNumber"):
            map_beg = int(beg_atom.GetProp("molAtomMapNumber"))
            map_end = int(end_atom.GetProp("molAtomMapNumber"))
            products_bonds.add(tuple(sorted((map_beg, map_end))))

    # 找出变化的键（反应中心）
    broken_bonds = reactants_bonds - products_bonds  # 断裂的键
    formed_bonds = products_bonds - reactants_bonds  # 形成的键
    reaction_center_bonds = broken_bonds.union(formed_bonds)

    # 5. 可视化
    print("断裂的键:", broken_bonds)
    print("形成的键:", formed_bonds)

    # 高亮显示反应中心原子
    reaction_center_atoms = set()
    for bond_pair in reaction_center_bonds:
        reaction_center_atoms.update(bond_pair)

    # 绘制反应物并高亮反应中心
    highlight_atoms = [atom.GetIdx() for atom in reactants.GetAtoms()
                       if atom.HasProp("molAtomMapNumber") and
                       int(atom.GetProp("molAtomMapNumber")) in reaction_center_atoms]

    highlight_bonds = []
    for bond in reactants.GetBonds():
        beg_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if beg_atom.HasProp("molAtomMapNumber") and end_atom.HasProp("molAtomMapNumber"):
            map_beg = int(beg_atom.GetProp("molAtomMapNumber"))
            map_end = int(end_atom.GetProp("molAtomMapNumber"))
            if tuple(sorted((map_beg, map_end))) in broken_bonds:
                highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(reactants, size=(800, 400),
                          highlightAtoms=highlight_atoms,
                          highlightBonds=highlight_bonds,
                          highlightColor=(0.7, 0.7, 1))  # 浅蓝色高亮

    # 添加说明文字
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"反应物: {reactants_smiles}\n反应模板: {reaction_smirks}\n"
                        f"反应中心原子: {reaction_center_atoms}\n"
                        f"断裂的键: {broken_bonds}\n形成的键: {formed_bonds}",
              fill="black")

    display(img)


# 示例使用
if __name__ == "__main__":
    # 示例1：酯化反应
    reactants = "CCO.C(=O)O"  # 乙醇 + 乙酸
    smirks = "[C:1][O:2][H:3].[C:4](=[O:5])[O:6][H:7]>>[C:1][O:2][C:4](=[O:5]).[H:3][O:6][H:7]"
    extract_reaction_center(reactants, smirks)

    # 示例2：Diels-Alder反应
    print("\n第二个示例:")
    reactants = "C1=CC=CC=C1.C=C"  # 苯 + 乙烯
    smirks = "[C:1]1=[C:2][C:3]=[C:4][C:5]=[C:6]1.[C:7]=[C:8]>>[C:1]12[C:2]=[C:3][C:4]=[C:5][C:6]1[C:7]2[C:8]"
    extract_reaction_center(reactants, smirks)