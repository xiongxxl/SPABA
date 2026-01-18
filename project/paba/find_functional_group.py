from rdkit import Chem
import pandas as pd
# 定义常见官能团的SMARTS模式及其化学式
# functional_groups = {
#     "Alkane": {"smarts": "[CX4]", "formula": "C-H"},
#     "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
#     "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
#     "Aromatic": {"smarts": "c1ccccc1", "formula": "C6H5"},
#     "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
#     "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
#     "Phenol": {"smarts": "c1ccc(O)cc1", "formula": "C6H5OH"},
#     "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
#     "Aldehyde": {"smarts": "[CX3H1](=O)[#6]", "formula": "R-CHO"},
#     "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
#     "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
#     "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
#     "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
#     "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
#     "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
#     "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
#     "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
#     "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
#     "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
#     "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
#     "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
#     "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
#     "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
#     "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
#     "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
#     "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
#     "Cyano":{"smarts": "[NX1]#[CX2]", "formula":"R-C≡N" },
#      "hydrazine":{"smarts":"[NH2]-[NH]" ,"formula":"R-NH-NH2"},
#                  }

functional_groups = {
    "Alkane": {"smarts": "[CX4]", "formula": "C-H"},
    "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
    "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
    "Aromatic": {"smarts": "c1ccccc1", "formula": "C6H5"},
    "Halide": {"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
    "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
    "Phenol": {"smarts": "c1ccc(O)cc1", "formula": "C6H5OH"},
    "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
    "Aldehyde": {"smarts": "[CX3H1](=O)[#6]", "formula": "R-CHO"},
    "Ketone": {"smarts": "[CX3](=O)[#6]", "formula": "R-CO-R"},
    "Carboxylic Acid": {"smarts": "[CX3](=O)[OX2H1]", "formula": "R-COOH"},
    "Ester": {"smarts": "[CX3](=O)[OX2][#6]", "formula": "R-COO-R"},
    "Amide": {"smarts": "[NX3][CX3](=[OX1])[#6]", "formula": "R-CONH2"},
    "Amine": {"smarts": "[NX3][#6]", "formula": "R-NH2"},
    "Nitrate": {"smarts": "[NX3](=O)([OX1-])[OX1-]", "formula": "R-NO3"},
    "Nitro": {"smarts": "[NX3](=O)[OX1-]", "formula": "R-NO2"},
    "Sulfonic Acid": {"smarts": "S(=O)(=O)[O-]", "formula": "R-SO3H"},
    "Thiol": {"smarts": "[SX2H]", "formula": "R-SH"},
    "Thioether": {"smarts": "[SX2][#6]", "formula": "R-S-R"},
    "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
    "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
    "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
    "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
    "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
    "Halide_a": {"smarts": "c-[F,Cl,Br,I]", "formula": "X- (F, Cl, Br, I)_a"},
    "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
    "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
    "Cyano":{"smarts": "[NX1]#[CX2]", "formula":"R-C≡N" },
    "Alcohol_a": {"smarts": "C-[OX2H]", "formula": "R-OH_a"},
     "Hydrazine":{"smarts": "[NH2]-[NH]" ,"formula": "R-NH-NH2"},
     "Hydroxylamine":{"smarts":"[NH2]-[OH]","formula": "R-NH-OH"},
     "Mg":{"smarts": "C-[Mg+]","formula": "R-Mg"},
     "Halide_b": {"smarts": "C-[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_b"}

}



def identify_functional_groups(frag_smiles,smiles):

        # 使用RDKit解析SMILES字符串
        mol = Chem.MolFromSmiles(frag_smiles)
        functional_groups_state = []
        functional_groups_axis=[]
        mol_data = {'smiles': smiles}
        for group_name, properties in functional_groups.items():
            pattern = Chem.MolFromSmarts(properties["smarts"])
            if mol is not None and hasattr(mol, 'HasSubstructMatch'):
                #if mol.HasSubstructMatch(pattern):
                mol_data[properties["formula"]] = mol.HasSubstructMatch(pattern)
        functional_groups_state.append(mol_data)
        #创建 pandas 数据框
        # df = pd.DataFrame(data)
        # # 保存到 Excel 文件
        # df.to_excel("functional_groups_test.xlsx", index=False)
        # print("官能团检测结果已保存到 functional_groups_test.xlsx")
        true_keys = [k for item in functional_groups_state for k, v in item.items() if v is True]
        if not true_keys:
            functional_groups_detail =frag_smiles
        else:
            ture_keys_str =','.join(true_keys)
            functional_groups_detail = ture_keys_str

        return functional_groups_state, functional_groups_detail



if __name__ == "__main__":
    # 示例输入：SMILES字符串
    #smiles = "CC(=O)O"  # 乙酸
    frag_smiles='BrCCBr.Sc1cccs1'
    smiles='BrCCBr.Sc1cccs1'

    # 识别并输出官能团
    functional_groups_state, functional_groups_detail = identify_functional_groups(frag_smiles,smiles)
    print(functional_groups_state)
    print(functional_groups_detail)
    # for group_name, formula in detected_groups:
    #     print(f"Detected functional group: {group_name}, Chemical formula: {formula}")


## function group chinese
# Alkane: "[CX4]" 匹配所有四价的碳原子，即饱和碳（烷烃）。
# Alkene: "[CX3]=[CX3]" 匹配双键碳原子（烯烃）。
# Alkyne: "[CX2]#[CX2]" 匹配三键碳原子（炔烃）。
# Aromatic: "c1ccccc1" 匹配苯环。
# Halide: "[F,Cl,Br,I]" 匹配卤素原子（氟、氯、溴、碘）。
# Alcohol: "[OX2H]" 匹配羟基（酒精）。
# Phenol: "c1ccc(O)cc1" 匹配苯酚。
# Ether: "OD2[#6]" 匹配醚。
# Aldehyde: "CX3H1[#6]" 匹配醛。
# Ketone: "CX3[#6]" 匹配酮。
# Carboxylic Acid: "CX3[OX2H1]" 匹配羧酸。
# Ester: "CX3[OX2][#6]" 匹配酯。
# Amide: "[NX3]CX3[#6]" 匹配酰胺。
# Amine: "[NX3][#6]" 匹配胺。
# Nitrate: "NX3([OX1-])[OX1-]" 匹配硝酸盐。
# Nitro: "NX3[OX1-]" 匹配硝基。
# Sulfonic Acid: "S(=O)(=O)[O-]" 匹配磺酸。
# Thiol: "[SX2H]" 匹配硫醇。
# Thioether: "[SX2][#6]" 匹配硫醚。
# Disulfide: "[SX2][SX2]" 匹配二硫化物。
# Sulfoxide: "SX3[#6]" 匹配亚砜。
# Sulfone: "SX4(=O)[#6]" 匹配砜。
# Phosphine: "[PX3]" 匹配膦。
# Phosphate: "P(=O)(O)(O)O" 匹配磷酸。
# Isocyanate: "N=C=O" 匹配异氰酸酯。
# Isothiocyanate: "N=C=S" 匹配异硫氰酸酯。

#za huai functional group

    # "Furan": {"smarts": "[#6]1=[#6][#6]=[#6][O]1", "formula": "R-C4H-O-R"},
    # "Pyrrole": {"smarts": "[#6]1=[#6][#6]=[#6][N]1", "formula": "R-C4H-N-R"},
    # "Thiophene": {"smarts": "[#6]1=[#6][#6]=[#6][S]1", "formula": "R-C4H-S-R"},
    # "Oxazole": {"smarts": "[#6]1=[N][#6]=[#6][O]1", "formula": "R-C3H-N-O-R"},
    # "Imidazole": {"smarts": "[#6]1=[N][#6]=[#6][N]1", "formula": "R-C3H-N2-R"},
    # "Thiazole": {"smarts": "[#6]1=[N][#6]=[#6][S]1", "formula": "R-C3H-N-S-R"},
    # "Isoxazole": {"smarts": "[N]1=[#6][#6]=[#6][O]1", "formula": "R-C3H-N-O-R"},
    # "Pyrazole": {"smarts": "[N]1=[#6][#6]=[#6][N]1", "formula": "R-C3H-N2-R"},
    # "Isothiazole": {"smarts": "[N]1=[#6][#6]=[#6][S]1", "formula": "R-C3H-N-S-R"},
    # "Pyridine": {"smarts": "[#6]1=[N][#6]=[#6][#6]=[#6]1", "formula": "R-C5H-N-R"},
    # "Pyran": {"smarts": "[#6]1=[#6][O][#6]=[#6][#6]1", "formula": "R-C5H-O-R"},
    # "Pyridazine": {"smarts": "[#6]1=[N][#6]=[N][#6]=[#6]1", "formula": "R-C4H-N2-R"},
    # "Pyrimidine": {"smarts": "[#6]1=[N][#6]=[#6][N]=[#6]1", "formula": "R-C4H-N2-R"},
    # "Pyrazine": {"smarts": "[#6]1=[N][#6]=[#6][#6]=[N]1", "formula": "R-C4H-N2-R"},
    # "γ-pyrone": {"smarts": "[O]=[#6]1[#6]=[#6][O][#6]=[#6]1", "formula": "R-C5H-O2-R"},
    # "α-pyrone": {"smarts": "[O]=[#6]1[O][#6]=[#6][#6]=[#6]1", "formula": "R-C5H-O2-R"},
    # "1H-1,2-diazepine": {"smarts": "[#6]1=[N][N][#6]=[#6][#6]=[#6]1", "formula": "R-C5H-N2-R"},
    # "1H-azepine": {"smarts": "[#6]1=[#6][N][#6]=[#6][#6]=[#6]1", "formula": "R-C6H-N-R"},
    # "2H-azepin-2-one": {"smarts": "[O]=[#6]1[#6]=[#6][#6]=[#6][N]=[#6]1", "formula": "R-C6H-N-O-R"},
    # "1,2-thiazepine": {"smarts": "[#6]1=[N][S][#6]=[#6][#6]=[#6]1", "formula": "R-C5H-N-S-R"},
    # "1,2-diazepane": {"smarts": "[#6]1[N][N][#6][#6][#6][#6]1", "formula": "R-C5H-N2-R"},
    # "Azocane": {"smarts": "[#6]1[#6][#6][#6][#6][N][#6][#6]1", "formula": "R-C7H-N-R"},
    # "azocan-2-one": {"smarts": "[O]=[#6]1[#6][#6][#6][#6][#6][N]1", "formula": "R-C7H-N-O-R"},
    # "1,5-diazocan-2-one": {"smarts": "[O]=[#6]1[#6][N][#6][#6][N][#6]1", "formula": "R-C6H-N2-O-R"},
    # "(Z)-1,2,3,4,7,8-hexahydroazocine": {"smarts": "[#6]1=[#6]\\[#6][#6][N][#6][#6]\\1", "formula": "R-C8H-N-R"},
    # "Indole": {"smarts": "[#6]1([N][#6]=[#6]2)=[#6]2[#6]=[#6][#6]=[#6]1", "formula": "R-C8H-N-R"},
    # "Purine": {"smarts": "[#6]1([N][#6]=[N]2)=[#6]2[#6]=[N][#6]=[#6]1", "formula": "R-C5H-N4-R"},
    # "Benzothiophene": {"smarts": "[#6]1([S][#6]=[#6]2)=[#6]2[#6]=[#6][#6]=[#6]1", "formula": "R-C8H-S-R"},
    # "Benzofuran": {"smarts": "[#6]1([O][#6]=[#6]2)=[#6]2[#6]=[#6][#6]=[#6]1", "formula": "R-C8H-O-R"},
    # "Benzopyran": {"smarts": "[#6]1([O][#6]=[#6]2)=[#6]2[#6]=[#6][#6]=[#6]1", "formula": "R-C9H-O-R"},
    # "Benzo-γ-pyrone": {"smarts": "[O]=[#6]1[#6]=[#6][O][#6]=[#6]2=[#6]1[#6]=[#6]=[#6]2", "formula": "R-C9H-O2-R"},
    # "Quinoline": {"smarts": "[#6]1([#6]=[#6][#6]=[#6]2)=[#6]2[N]=[#6]=[#6]1", "formula": "R-C9H-N-R"},
    # "Isoquinoline": {"smarts": "[#6]1([#6]=[#6][#6]=[#6]2)=[#6]2[#6]=[N]=[#6]1", "formula": "R-C9H-N-R"},
    # "2,3,4,5-tetrahydro-1H-benzo[b]azepine": {"smarts": "[#6]12=[#6][#6]=[#6][#6]=[#6]1[N][#6][#6][#6][#6]2", "formula": "R-C11H-N-R"},
    # "1H-benzo[b]azepine": {"smarts": "[#6]12=[#6][#6]=[#6][#6]=[#6]1[N]=[#6][#6]=[#6]2", "formula": "R-C11H-N-R"},
    # "benzo[f][1,2]thiazepine": {"smarts": "[#6]12=[#6][#6]=[#6][#6]=[#6]1[S][N]=[#6]=[#6]2", "formula": "R-C10H-N-S-R"},
    # "(1Z,5Z)-4,6a-dihydro-3H-pyrido[1,2-d][1,4]diazocine": {"smarts": "[#6]12[#6]=[#6][#6]=[#6][N]1/\\[#6]=[N]/[#6]=[N]/[#6]=[#6]2", "formula": "R-C9H-N3-R"},
    # "Carbazole": {"smarts": "[#6]1([N]2)=[#6]([#6]3=[#6]2[#6]=[#6][#6]=[#6]3)[#6]=[#6]=[#6]1", "formula": "R-C12H-N-R"},
    # "Acridine": {"smarts": "[#6]12=[#6][#6]=[#6][#6]=[#6]1[#6]=[#6]3([#6][#6]=[#6][#6]=[#6]3)=N2", "formula": "R-C13H-N-R"}