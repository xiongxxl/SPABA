from chemical_atom_number import get_atom_indices_by_symbol
from highlight_num_atoms import highlight_num_atoms
from rdkit import Chem
from annotate_num_chemical import annotate_atoms
from rdkit import Chem
from divide_by_dot_criterion import divide_by_dot_criterion
def standardize_single_smiles(smiles):
    """
    将单个 SMILES 转换为标准化 SMILES，无法解析时返回原始 SMILES。
    Args:
        smiles (str): 输入的 SMILES 字符串。
    Returns:
        str: 标准化后的 SMILES，或者原始 SMILES（如果无法解析）。
    """
    try:
        # 转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 转换为标准化的 SMILES
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            # 如果无法解析，返回原始 SMILES
            return smiles
    except Exception as e:
        # 捕获异常并返回原始 SMILES
        print(f"Error processing SMILES {smiles}: {e}")
        return smiles
# 示例 SMILES
# raw_smiles = "C1=CC=CC=C1"  # 这是苯的SMILES
# standardized_smiles = standardize_single_smiles(raw_smiles)
# print(f"Input SMILES: {raw_smiles}")
# print(f"Standardized SMILES: {standardized_smiles}")

# # 示例 SMILES 输入
# input_smiles = ["C1=CC=CC=C1", "CC(C)C(=O)O", "C#CCBr", "INVALID_SMILES"]
# output_smiles = standardize_smiles(input_smiles)

def contains_fragment(smiles):
    # 将SMILES转换为分子对象
    smiles=standardize_single_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # 如果SMILES无效，则返回False

    # 遍历分子中的所有环
    for ring in mol.GetRingInfo().AtomRings():

        if len(ring) == 6:

            atoms_in_ring = [mol.GetAtomWithIdx(i) for i in ring]

            if all(atom.GetIsAromatic() for atom in atoms_in_ring):
                return True
    return False


# # 测试样例
# smiles_1 = "C1=CC=CC=C1"
# smiles_2 = "N#CC1=CC=CS1.OC"  #
# print(contains_benzene(smiles_1))  # 输出：True
# print(contains_benzene(smiles_2))  # 输出：False


def retrain_fragment_indices(smiles,frag_indices):
    smiles = standardize_single_smiles(smiles)
    if contains_fragment(smiles):
        mol = Chem.MolFromSmiles(smiles)
        benzene_ring = Chem.MolFromSmiles('c1ccccc1')
        benzene_matches =list(mol.GetSubstructMatches(benzene_ring)[0])
        frag_indices_del_fragment= [x for x in frag_indices if x not in benzene_matches]
    else:
        frag_indices_del_fragment=frag_indices
    return frag_indices_del_fragment


def subset_to_main(A,B):
    # 遍历B中的每个子列表
    for sublist in B:
        # 判断A中是否包含子列表的元素
        if all(item in A for item in sublist):
            # 如果包含，将该子列表赋值给A
            A = sublist
    return A

    # print(A)
    # A = [9, 8, 10]
    # B = [[7, 8], [8, 9]]


def restrain_functional_axis(smiles, reactive_atoms):
    #reactive_atoms=retrain_fragment_indices(smiles,reactive_atoms)
    mol = Chem.MolFromSmiles(smiles)
     # 官能团集合
    functional_groups = {
         "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
         "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
         "Halide":{"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
         "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
         "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
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
          "Alcohol_c": {"smarts": "[c] - [OX2H]", "formula": "R - OH_c"},
         "Disulfide": {"smarts": "[SX2][SX2]", "formula": "R-S-S-R"},
         "Sulfoxide": {"smarts": "[SX3](=O)[#6]", "formula": "R-S(=O)-R"},
         "Sulfone": {"smarts": "[SX4](=O)(=O)[#6]", "formula": "R-SO2-R"},
         "Phosphine": {"smarts": "[PX3]", "formula": "R3P"},
         "Halide_C": {"smarts": "[C]-[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_C"},
         "Phosphate": {"smarts": "P(=O)(O)(O)O", "formula": "R-O-PO3H2"},
         "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
         "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
         "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
         "hydrazine":{"smarts":"[NH2]-[NH]" ,"formula":"R-NH-NH2"},
         "Amide_a" :{"smarts":"NX3CX3#6","formula":"R-CO-NH"},
         "Halide_c": {"smarts": "[c]-[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_c"},
         "Alcohol_C": {"smarts": "[C]-[OX2H]", "formula": "R-OH_C"},
         "Nitroso":{"smarts": "[NX2]=O","formula":"R-N=O"},


    }

#   # 存储结果的列表
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

    #del excessive atoms
    #restrain R-O-R
    if any(item[0] == 'R-COOH' for item in results) and any(item[0] == 'R-CO-R' for item in results):
        results = [item for item in results if item[0] != 'R-CO-R']

    atom_indices_R_O_R=get_atom_indices_by_symbol(smiles, 'O')
    results_restrain_1 = [(item[0], tuple(x for x in item[1] if x == atom_indices_R_O_R)) if item[0] == 'R-O-R' else item
                           for item in results
                         ]

    # restrain CO
    atom_indices_CO = get_atom_indices_by_symbol(smiles, 'O')
    atom_indices_CO_multi =([[atom_indices_CO[i] - 1, atom_indices_CO[i]] for i in range(len(atom_indices_CO))] +
                            [[atom_indices_CO[i], atom_indices_CO[i] + 1] for i in range(len(atom_indices_CO))])
    results_restrain_2 = [(item[0], tuple(subset_to_main(item[1],atom_indices_CO_multi))) if item[0] == 'R-CO-R' else item
                         for item in results_restrain_1
                         ]
    #restrain R-NH2
    atom_indices_NH2 = get_atom_indices_by_symbol(smiles, 'N')
    atom_indices_NH2_multi=[[x] for x in atom_indices_NH2 ]
    results_restrain_3 = [(item[0], tuple(subset_to_main(item[1],atom_indices_NH2_multi))) if item[0] == 'R-NH2' else item
                           for item in results_restrain_2
                         ]

    #exact the part of number
    results_del_min_2_number = tuple(item[1] for item in results_restrain_3)
    functional_arrays=results_del_min_2_number

    for fucntional_array in functional_arrays:
        if set(reactive_atoms)&set(fucntional_array):
            reactive_atoms_restrain.extend(fucntional_array)
    reactive_atoms_restrain_result = list(set(reactive_atoms_restrain) | set(reactive_atoms))
    #reactive_atoms_restrain_result =reactive_atoms_restrain

    #jude functional group own to before or after dot
    annotated_smiles, img, numbers_before_dot, numbers_after_dot=annotate_atoms(smiles)
    dot_criterion=min(max(numbers_before_dot),max(numbers_after_dot))
    less_dot_criterion, greater_equal_dot_criterion = divide_by_dot_criterion(dot_criterion, functional_arrays)
    less_dot_criterion_list = [item for sublist in less_dot_criterion for item in sublist]
    greater_equal_dot_criterion_list = [item for sublist in greater_equal_dot_criterion for item in sublist]

    return results,reactive_atoms_restrain_result,less_dot_criterion_list, greater_equal_dot_criterion_list



def restrain_functional_axis_single(smiles, reactive_atoms):
    #reactive_atoms=retrain_fragment_indices(smiles,reactive_atoms)
    mol = Chem.MolFromSmiles(smiles)
     # 官能团集合
    functional_groups = {
         "Alkene": {"smarts": "[CX3]=[CX3]", "formula": "C=C"},
         "Alkyne": {"smarts": "[CX2]#[CX2]", "formula": "C≡C"},
         "Halide_1":{"smarts": "[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)_1"},
         "Alcohol": {"smarts": "[OX2H]", "formula": "R-OH"},
         "Ether": {"smarts": "[OD2]([#6])[#6]", "formula": "R-O-R"},
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
         "Isocyanate": {"smarts": "N=C=O", "formula": "R-N=C=O"},
         "Isothiocyanate": {"smarts": "N=C=S", "formula": "R-N=C=S"},
         "Cyano": {"smarts": "[NX1]#[CX2]", "formula": "R-C≡N"},
         "hydrazine":{"smarts":"[NH2]-[NH]" ,"formula":"R-NH-NH2"},
         #" Amide" :{"smarts":"NX3CX3#6","formula":"R-CO-NH"},
         "Halide": {"smarts": "[c]-[F,Cl,Br,I]", "formula": "X (F, Cl, Br, I)"},
         "Alcohol_1": {"smarts": "[C]-[OX2H]", "formula": "R-OH_1"},
    }

#   # 存储结果的列表
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

    #del excessive atoms
    #restrain R-O-R
    if any(item[0] == 'R-COOH' for item in results) and any(item[0] == 'R-CO-R' for item in results):
        results = [item for item in results if item[0] != 'R-CO-R']

    atom_indices_R_O_R=get_atom_indices_by_symbol(smiles, 'O')
    results_restrain_1 = [(item[0], tuple(x for x in item[1] if x == atom_indices_R_O_R)) if item[0] == 'R-O-R' else item
                           for item in results
                         ]

    # restrain CO
    atom_indices_CO = get_atom_indices_by_symbol(smiles, 'O')
    atom_indices_CO_multi =([[atom_indices_CO[i] - 1, atom_indices_CO[i]] for i in range(len(atom_indices_CO))] +
                            [[atom_indices_CO[i], atom_indices_CO[i] + 1] for i in range(len(atom_indices_CO))])
    results_restrain_2 = [(item[0], tuple(subset_to_main(item[1],atom_indices_CO_multi))) if item[0] == 'R-CO-R' else item
                         for item in results_restrain_1
                         ]
    #restrain R-NH2
    atom_indices_NH2 = get_atom_indices_by_symbol(smiles, 'N')
    atom_indices_NH2_multi=[[x] for x in atom_indices_NH2 ]
    results_restrain_3 = [(item[0], tuple(subset_to_main(item[1],atom_indices_NH2_multi))) if item[0] == 'R-NH2' else item
                           for item in results_restrain_2
                         ]

    #exact the part of number
    results_del_min_2_number = tuple(item[1] for item in results_restrain_3)
    functional_arrays=results_del_min_2_number

    for fucntional_array in functional_arrays:
        if set(reactive_atoms)&set(fucntional_array):
            reactive_atoms_restrain.extend(fucntional_array)
    reactive_atoms_restrain_result = list(set(reactive_atoms_restrain) | set(reactive_atoms))
    #reactive_atoms_restrain_result =reactive_atoms_restrain



    return results,reactive_atoms_restrain_result


if __name__ == "__main__":
    location_restrain=[]
    smiles="NC1=CC(F)=CC=C1I.C#CC2=CC=CC=C2"
    reactive_atoms=[4,5,6,7]
    results,reactive_atoms_restrain_result,less_dot_criterion, greater_equal_dot=restrain_functional_axis(smiles, reactive_atoms)

    # print('fucntional_arrays:',results)
    # print('fucntional_arrays:',less_dot_criterion)
    # print('fucntional_arrays:', greater_equal_dot)

    #print('functional_dict:',functional_dict)
    # for fucntional_array in results:
    #     if set(location_del_benzene)&set(fucntional_array):
    #         location_restrain.extend(fucntional_array)
    #         print('location_restrain',location_restrain)

    # img_data=highlight_num_atoms(smiles, location_restrain)
    # #保存图像为文件
    # # with open("highlighted_molecule.png", "wb") as f:
    # #   f.write(img_data)
    # # #显示图像
    # from PIL import Image
    # import io
    # img = Image.open(io.BytesIO(img_data))
    # img.show()



