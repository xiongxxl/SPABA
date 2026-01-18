The toxicity in honey bees (beet) dataset was extract from a study on the prediction of acute contact toxicity of pesticides in honeybees.\cite{venko2018classification}. The data set contains 254 compounds with their experimental values. Two sets binary labels were provided. For the first, the compound with experimental values up to 1 mu/bee was considered positive otherwise negative. For the second set, using the threshold 100 mu/bee, the compound with experimental values up to 100 mu/bee was considered positive otherwise negative.

The data file contains a csv table, in which columns below are used:
    - "SMILES" - SMILES representation of the molecular structure
    - "Experimental value" - The mortality of honey bees is recorded after 48 h of exposure and results are presented in terms of mg active substance/bee as the median lethal dose (LD50 active substance/bee, μg/bee).
    - "threshold_1" - The compound with experimental values up to 1 mu/bee was considered positive otherwise negative.
    - "threshold_100" - The compound with experimental values up to 100 mu/bee was considered positive otherwise negative.

References:
@article{venko2018classification,
  title={Classification models for identifying substances exhibiting acute contact toxicity in honeybees (Apis mellifera) $},
  author={Venko, K and Drgan, V and Novi{\v{c}}, M},
  journal={SAR and QSAR in Environmental Research},
  volume={29},
  number={9},
  pages={743--754},
  year={2018},
  publisher={Taylor \& Francis}
}
