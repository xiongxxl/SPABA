from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np

def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        fps.append(fp)
    return fps

def cluster_fingerprints(fps, cutoff=0.3):
    # 计算距离矩阵，距离=1 - 相似度
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        for j in range(i):
            if fps[i] is None or fps[j] is None:
                dist = 1.0
            else:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dist = 1 - sim
            dists.append(dist)
    # Butina聚类，cutoff表示最大距离阈值，阈值越小聚类越细
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return clusters

if __name__ == '__main__':
    smiles_list = [
        "CCO", "CCN", "CCC", "CCCl", "C1CCCCC1", "c1ccccc1", "CC(=O)O", "CC(=O)N", "CCCN"
    ]
    fps = smiles_to_fingerprint(smiles_list)
    clusters = cluster_fingerprints(fps, cutoff=0.3)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}:")
        for idx in cluster:
            print(f"  {smiles_list[idx]}")