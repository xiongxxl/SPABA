import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import time



def set_random_seed(random_seed):
    random.seed(random_seed)

def unique_list(list_):
    a = sorted(list(set(list_)))
    random.shuffle(a)
    return a

def match_hydroxyl(smiles):
    reaction_add_protecting_group = AllChem.ReactionFromSmarts('[C:1](=[O])-[OH]>>[C:1](=[O])-[AsH]-[AsH2]')

    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_protecting_group = copy.deepcopy(molecule)
    while molecule_with_protecting_group.HasSubstructMatch(Chem.MolFromSmarts('[C](=[O])-[OH]')):
        molecule_with_protecting_group = reaction_add_protecting_group.RunReactants((molecule_with_protecting_group,))[0][0]
        Chem.SanitizeMol(molecule_with_protecting_group)

    hydroxyl_count = len(molecule_with_protecting_group.GetSubstructMatches(Chem.MolFromSmarts('[C]-[OH]')))
    return hydroxyl_count
    
    # if you want to run one reaction only match hydroxyl but not match carboxyl, use the code below 
    reaction_remove_protecting_group = AllChem.ReactionFromSmarts('[C:1](=[O])-[AsH]-[AsH2]>>[C:1](=[O])-[OH]')
    reaction = AllChem.ReactionFromSmarts('[c:1]-[OH]>>[c:1]-[O]-[CH](-[F])-[F]')
    result = reaction.RunReactants((molecule_with_protecting_group,))
    product_smiles_list = []
    for temp_product_list in result:
        molecule_without_protecting_group = temp_product_list[0]
        Chem.SanitizeMol(molecule_without_protecting_group)
        while molecule_without_protecting_group.HasSubstructMatch(Chem.MolFromSmarts('[C](=[O])-[AsH]-[AsH2]')):
            molecule_without_protecting_group = reaction_remove_protecting_group.RunReactants((molecule_without_protecting_group,))[0][0]
            Chem.SanitizeMol(molecule_without_protecting_group)
        product_smiles = Chem.MolToSmiles(molecule_without_protecting_group)
        product_smiles_list.append(product_smiles)
    product_smiles_list = unique_list(product_smiles_list)

    return hydroxyl_count, product_smiles_list


# [C]=[C] 双键
# [C]#[C]
# [F,Cl,Br,I] [F]
# [OH] 包含羧基的羟基 
# [NH2] 氨基 也包含酰胺
# [C,c:1]-[O]-[C,c:2] 醚 小c是芳基
# [C,c:1]-[C](=[O])-[C,c:2] 酮
# [C](=[O])-[OH]
# [C,c:1]-[C](=[O])-[O]-[C,c:2] 酯
# [C,c:1]-[C](=[O])-[NH]-[C,c:2] 酰胺
# [N+](=[O])[O-]
# [NH2]-[NH2]
# [C,c:1]-[NH]-[NH2] 肼
# [S](=[O])(=[O])-[OH] 磺酸
# [C,c:1]-[SH]
# [C,c:1]-[S]-[S]-[C,c:1] 
# [C,c:1]-[S](=[O])(=[O])-[C,c:2] 砜
# [C,c:1]-[S](=[O])-[C,c:2] 亚砜
# [C,c:1]-[S]-[C,c:2] 醚 小c是芳基
# [C,c:1]-[CH](=[O]) 醛
# [C,c:1]-[C]#[N] 腈
# [C,c:1]-[N]=[C]=[S] 异硫氰酸酯
# [C,c:1]-[N]=[C]=[O] 异氰酸酯
# [C,c:1]-[O]-[P](=[O])(-[O]-[C,c:2])-[OH] 磷酸酯
# [P](-[C,c:1])(-[C,c:1])-[C,c:1] 膦



if __name__ == "__main__":
    set_random_seed(1024)
    start_time = time.time()
    smiles = 'OCC1=C(CC(O)=O)C(C(O)=O)=C(C=C1O)C1=CC(=CC=C1[AsH2])C(O)=O'
    print("count of hydroxyl groups: ")
    print("Rdkit:", len(Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts('[C]-[OH]'))))
    print("Ours:", match_hydroxyl(smiles))
    print(f'Finished in {round(time.time() - start_time, 3)} second(s)')