from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import Draw

def highlight_num_atoms(smiles, atom_indices):
    """
    根据给定的原子索引高亮 SMILES 表示的分子中的原子。
    :param smiles: 分子的 SMILES 表示
    :param atom_indices: 需要高亮的原子索引列表
    """
    # 使用 RDKit 从 SMILES 创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    # highlight_colors={0:(0,0,255),1:(0,0,255)}

    # 创建分子绘图对象，并高亮指定的原子
    drawer = Draw.MolDraw2DCairo(300, 300)
    opts = drawer.drawOptions()
    drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
    # drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()
    return img

if __name__ == "__main__":
    # highlight atom
    #smiles="ClC1=CC=CC=C1I.C1(NC2=CC=CC=C2)=CC=CC=C1"
    #smiles="ClC1=CC=CC=C1I.C2(NC9=CC=CC=C9)=CC=CC=C2"
    #smiles="CC(C)C(=O)CC(=O)C(F)(F)F.Nc1ccccc1N"
    smiles="CO/N=C1CCCCC/1.CC(C)=O"
    location=[0,1]

    atom_indices=location
    # smiles="CCC(=O)CC(=O)C(F)(F)F.Nc1cccc

    # atom_indices=[0,3,6,16,17,18,11]
    img_data=highlight_num_atoms(smiles, atom_indices)
    #保存图像为文件
    # with open("highlighted_molecule.png", "wb") as f:
    #   f.write(img_data)
    # #显示图像
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_data))
    img.show()