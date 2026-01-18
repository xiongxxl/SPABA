import pandas as pd

# 1. 读入两个表
df_test = pd.read_excel("network_transformer_mcc_0.71_metrics_test.xlsx")
df_all  = pd.read_excel("uspto_yang_reactive_atom_orgin_50k_indices_atoms_deep.xlsx")

# 2. 只保留 df_test 里的 reactant 列，用它去匹配 df_all
#    用 merge 可以保证结果按照 df_test 的 reactant 顺序排列
df_sample = df_test[['reactant']].merge(
    df_all,
    on='reactant',
    how='left'   # 如果有匹配不到的 reactant，会出现 NaN，方便你检查
)

# 3. 保存为新的 excel 文件
df_sample.to_excel("network_transformer_mcc_0.71_metrics_test_sample.xlsx", index=False)

print("已生成：network_transformer_mcc_0.71_metrics_test_sample.xlsx")


