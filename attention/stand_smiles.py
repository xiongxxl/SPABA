from rdkit import Chem

smiles = "CC(=O)Nc1ccccc1C(N)=O"
mol = Chem.MolFromSmiles(smiles)
can_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
print(can_smiles)






