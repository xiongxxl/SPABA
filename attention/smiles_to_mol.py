# from rdkit import Chem
#
# # 输入SMILES字符串
# smiles = "CCO"  # 乙醇的SMILES表示法
#
# # 将SMILES字符串转换为分子对象
# mol = Chem.MolFromSmiles(smiles)
#
# # 打印分子的原子和键信息
# print("原子:")
# for atom in mol.GetAtoms():
#     print(atom.GetSymbol())
#
# print("\n键:")
# for bond in mol.GetBonds():
#     print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

print("欢迎使用化学式绘制工具！")
while 1:
    # 绘制主链
    print("示例：CCC(C)CC 是第三个碳原子上有有一个碳链，这个碳链由一个碳原子构成。")
    model = input("输入化学式：")
    if model == "exit":
        exit()
    # mol = Chem.MolFromSmiles('CCC(C)CC(C)C(C)CC')
    mol = Chem.MolFromSmiles(model)

    # 设定高度和宽度
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 200)

    # 设置绘制选项
    opt = drawer.drawOptions()
    opt.additionalAtomLabelPadding = 0.2

    # 绘制化学结构式
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # 将绘制结果保存为SVG格式文件
    with open('output.svg', 'w') as f:
        f.write(drawer.GetDrawingText())
    print("图片已经生成，请到目录下查看！")