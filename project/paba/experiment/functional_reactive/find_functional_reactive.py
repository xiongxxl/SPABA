
from highlight_num_atoms import highlight_num_atoms
from rdkit import Chem

def find_functional_axis(smiles):
    #reactive_atoms=retrain_fragment_indices(smiles,reactive_atoms)
    mol = Chem.MolFromSmiles(smiles)

     # 官能团集合
    # functional_groups = {
    #      "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
    #      "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
    #      "Halide_1":{"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_1"},
    #      "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
    #      "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
    #      "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
    #      "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
    #      "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
    #      "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
    #      "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
    #      "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
    #      "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
    #      "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
    #      "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
    #      "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
    #      "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
    #      "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
    #      "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
    #      "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
    #      "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
    #      "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
    #      "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
    #      "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
    #      "hydrazine":{"smarts":"[NH2]-[NH]" ,"formula":"R-NH-NH2"},
    #      #" Amide" :{"smarts":"NX3CX3#6","formula":"R-CO-NH"},
    # }

     #官能团集合 by tianyang
    functional_groups = {
         "Alkene": {"smarts": "[C]=[C]", "formula": "C=C"},
         "Alkyne": {"smarts": "[C]#[C]", "formula": "C≡C"},
         "Halide_1":{"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_1"},
         "Alcohol": {"smarts": "[OH]", "formula": "R-OH"},
         "Ether": {"smarts": "[C,c:1]-[O]-[C,c:2]", "formula": "R-O-R"},
         "Ketone": {"smarts": "[C,c:1]-[C](=[O])-[C,c:2]", "formula": "R-CO-R"},
         "Ketone_1": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R_1"},
         "Carboxylic Acid": {"smarts": "[C](=[O])-[OH]", "formula": "R-COOH"},
         "Ester": {"smarts": "[C,c:1]-[C](=[O])-[O]-[C,c:2]", "formula": "R-COO-R"},
         "Amide": {"smarts": "[C,c:1]-[C](=[O])-[NH]-[C,c:2]", "formula": "R-CONH2"},
         "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
         "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
         "Nitro": {"smarts": "[N+](=[O])[O-]", "formula": "R-NO2"},
         "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
         "Thiol": {"smarts": "[C,c:1]-[SH]", "formula": "R-SH"},
         "Thioether": {"smarts": "[C,c:1]-[S]-[C,c:2]", "formula": "R-S-R"},
         "Disulfide": {"smarts": "[C,c:1]-[S]-[S]-[C,c:1]", "formula": "R-S-S-R"},
         "Sulfoxide": {"smarts": "[C,c:1]-[S](=[O])-[C,c:2]", "formula": "R-S(=O)-R"},
         "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
         "Phosphine": {"smarts": "[P](-[C,c:1])(-[C,c:1])-[C,c:1]", "formula": "R3P"},
         "Phosphate": {"smarts": "[C,c:1]-[O]-[P](=[O])(-[O]-[C,c:2])-[OH]", "formula": "R-O-PO3H2"},
         "Isocyanate": {"smarts": "[C,c:1]-[N]=[C]=[O]", "formula": "R-N=C=O"},
         "Isothiocyanate": {"smarts": "[C,c:1]-[N]=[C]=[S]", "formula": "R-N=C=S"},
         "Cyano": {"smarts": "[C,c:1]-[C]#[N]", "formula": "R-C≡N"},
         "hydrazine":{"smarts":"[C,c:1]-[NH]-[NH2]" ,"formula":"R-NH-NH2"},
         #" Amide" :{"smarts":"NX3CX3#6","formula":"R-CO-NH"},
    }


#     # 存储结果的列表
    results = []
    reactive_atoms_restrain=[]
    # 遍历所有官能团，查找匹配项
    for group_name, group_info in functional_groups.items():
        smarts_pattern = group_info["smarts"]
        pattern = Chem.MolFromSmarts(smarts_pattern)
        matches = mol.GetSubstructMatches(pattern)

        # 如果有匹配项，则记录官能团的公式和索引
        if matches:
            for match in matches:
                results.append((group_info["formula"], match))

    result_axis = [num for item in results for num in item[1]]
    return result_axis
#
# if __name__ == "__main__":
#     smiles='CC(C)C(=O)CC(=O)C(F)(F)F'
#     results,functional_arrays=find_functional_axis(smiles)
#     print(functional_arrays)

if __name__ == "__main__":
    location_restrain=[]
    smiles="IC1=CC=CC=C1.C2=CC=CN2"

    result_axis=find_functional_axis(smiles)

    img_data=highlight_num_atoms(smiles, result_axis)
    #保存图像为文件
    # with open("highlighted_molecule.png", "wb") as f:
    #   f.write(img_data)
    # #显示图像
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_data))
    img.show()



