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

if __name__ == "__main__":
    set_random_seed(1024)
    start_time = time.time()
    smiles = 'OCC1=C(CC(O)=O)C(C(O)=O)=C(C=C1O)C1=CC(=CC=C1[AsH2])C(O)=O'
    print("count of hydroxyl groups: ")
    print("Rdkit:", len(Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts('[C]-[OH]'))))
    print("Ours:", match_hydroxyl(smiles))
    print(f'Finished in {round(time.time() - start_time, 3)} second(s)')