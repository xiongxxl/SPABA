from rdkit import Chem
from rdkit.Chem import Draw
import re

def annotate_atoms(smiles):
    numbers_after_dot=[]
    numbers_before_dot=[]
    # 解析 SMILES
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        #atom.SetAtomMapNum(atom.GetIdx())
    # 生成带有原子编号的 SMILES
    annotated_smiles = Chem.MolToSmiles(mol)
    print(annotated_smiles)
        # 生成化学式图片
    img = Draw.MolToImage(mol, legend=annotated_smiles)



    return  img
if __name__ == "__main__":
    smiles="CCC(CC)=O.C1=NC=CN=N1"


    img = annotate_atoms(smiles)
    img.show()