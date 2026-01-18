This note provides the data and code of  this paper

![overview](overview.jpg)

Configurating environment

---

Python:3.12.1
Numpy 1.26.4
pandas:2.2.1
rdkit:2023.09.6
pytorch:2.2.2
seaborn :0.13.2
scipy:1.1.3.2

Calculating  of attention matrices

---

1. Download the pre-trained network parameter“chembl_pubchem_zinc_models” and place them in the **attention** folder. For details, refer to：[https://github.com/WeilabMSU/PretrainModels/blob/main/README.md]
2. We modified `torch.nn.functional` to save the 8×8 QKT matrix  since the original code lacks this interface—please copy our code entirely and ensure `torch.nn.functional` hasn't been overwritten after installing `torch`.
3. Convert the sample to smi format, place it in the `/data/input_smiles` folder, and run `generate_bt_fps.py` to save the resulting attention matrix in the `middle_attention` folder.


Extracting molecular fragments

---

1. After generating attention matrices for 10,000 molecules, run `/interpret/function/main_function_10000` to save the functional group statistics in the `statistics_function` folder and the images in `img_function`.
2. `divide_by_location_number` classifies the data based on the number of atoms, while `deal_statistic_data` calculates the frequency of functional group occurrences.
3. The `statistic_img` code visualizes the statistical results.


Predicting  reaction centers

---
1. According to the PABA algorithm, the reaction center attention matrix is obtained, and the ground truth of reaction sites, named `double_criterion_100`, is placed in the `statistic_reactive` folder.
2. `main_predict_heads` identifies the optimal heads, `main_find_alpha` determines the specific alpha values, `main_combine_heads` generates the combined results, and `main_predict_heads` provides the final prediction outcomes.
3. The `statistic_image` code displays both the data processing steps and the final results.






