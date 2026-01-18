from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io


# def highlight_num_atoms(smiles, atom_indices):
#     """
#     根据给定的原子索引高亮 SMILES 表示的分子中的原子。
#     :param smiles: 分子的 SMILES 表示
#     :param atom_indices: 需要高亮的原子索引列表
#     """
#     # 使用 RDKit 从 SMILES 创建分子对象
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         raise ValueError("Invalid SMILES string")
#
#     # highlight_colors={0:(0,0,255),1:(0,0,255)}
#
#     # 创建分子绘图对象，并高亮指定的原子
#     drawer = Draw.MolDraw2DCairo(300, 300)
#     opts = drawer.drawOptions()
#     drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
#     # drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
#     drawer.FinishDrawing()
#     img = drawer.GetDrawingText()
#     return img
#
# if __name__ == "__main__":
#     # highlight atom
#     #smiles="ClC1=CC=CC=C1I.C1(NC2=CC=CC=C2)=CC=CC=C1"
#     smiles="C=CCC(CC#CCC(CC#CC)C(OCC)=O)(C(OCC)=O)C(OCC)=O"
#     location=[15, 16, 20, 21]
#     atom_indices=location
#     img_data=highlight_num_atoms(smiles, atom_indices)
#     img = Image.open(io.BytesIO(img_data))
#     img.show()

import io
import os
import ast

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


def highlight_num_atoms(smiles, atom_indices):
    """
    根据给定的原子索引高亮 SMILES 表示的分子中的原子。
    :param smiles: 分子的 SMILES 表示
    :param atom_indices: 需要高亮的原子索引列表 (基于 RDKit atom index)
    :return: PNG 图片的二进制数据
    """
    # 使用 RDKit 从 SMILES 创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 创建分子绘图对象，并高亮指定的原子
    drawer = Draw.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()  # PNG 的二进制数据
    return img_bytes


def smiles_excel_to_images(
    excel_in,
    excel_out,
    img_dir="atom_images",
    smiles_col="smiles",
    atom_idx_col="atom_indices",
    img_col="atom_img",
):
    """
    读取一个 Excel，使用 highlight_num_atoms 生成高亮图片，
    将图片保存到本地，并把图片路径写入到 Excel 的 img_col 列。

    :param excel_in: 输入 Excel 路径
    :param excel_out: 输出 Excel 路径
    :param img_dir: 保存图片的文件夹
    :param smiles_col: SMILES 所在列名
    :param atom_idx_col: 原子索引所在列名（如 "[1, 2, 3]" 这种字符串）
    :param img_col: 要写入图片路径的列名
    """
    os.makedirs(img_dir, exist_ok=True)

    df = pd.read_excel(excel_in)

    img_paths = []

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col, None)
        atom_indices_raw = row.get(atom_idx_col, None)

        # 遇到缺失值就跳过
        if pd.isna(smiles) or pd.isna(atom_indices_raw):
            img_paths.append(None)
            continue

        # 解析 atom_indices：一般是字符串 "[15, 16, 20, 21]" 这种
        try:
            if isinstance(atom_indices_raw, str):
                atom_indices = ast.literal_eval(atom_indices_raw)
            elif isinstance(atom_indices_raw, (list, tuple)):
                atom_indices = list(atom_indices_raw)
            else:
                # 单个数字的情况
                atom_indices = [int(atom_indices_raw)]
        except Exception as e:
            print(f"Row {idx}: parse atom_indices error: {atom_indices_raw} ({e})")
            img_paths.append(None)
            continue

        try:
            img_bytes = highlight_num_atoms(smiles, atom_indices)
            img = Image.open(io.BytesIO(img_bytes))

            img_path = os.path.join(img_dir, f"row_{idx}.png")
            img.save(img_path)
            img_paths.append(img_path)
        except Exception as e:
            print(f"Row {idx}: draw/save error: {e}")
            img_paths.append(None)

    # 把图片路径写入新列
    df[img_col] = img_paths
    df.to_excel(excel_out, index=False)
    print(f"Done. Saved result to {excel_out}, images in {img_dir}/")


if __name__ == "__main__":
    # # 单例测试
    smiles = "OC1=CC=CC=C1.BrCCCC"
    location = [0,1,7,8]
    img_data = highlight_num_atoms(smiles, location)
    img = Image.open(io.BytesIO(img_data))
    img.show()

    # 批量处理 Excel
    # 假设你的 Excel 有两列： "smiles" 和 "atom_indices"
    # atom_indices 列的格式类似 "[15, 16, 20, 21]"
    # excel_in = "input_smiles_atoms.xlsx"
    # excel_out = "output_with_images.xlsx"
    #
    # smiles_excel_to_images(
    #     excel_in=excel_in,
    #     excel_out=excel_out,
    #     img_dir="atom_images",
    #     smiles_col="smiles",
    #     atom_idx_col="atom_indices",
    #     img_col="atom_img",
    # )
    #



