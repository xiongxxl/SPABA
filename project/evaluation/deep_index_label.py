
import re
import ast
import math
import pandas as pd
import os
# =========================
# 可配置：Excel 里的列名
# =========================
SMILES_COL = "smiles"            # 存 SMILES 的列
PRED_COL   = "predicted_atoms"   # 存 二值列表 字符串 的列
OUT_COL    = "atom_index"        # 输出的原子序号列名


# =========================
# tokenize_smiles
# =========================
SPACE_NORMALIZER = re.compile(r"\s+")

SMI_SYMBOLS = r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|" + \
                r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"

SMI_REGEX = re.compile(SMI_SYMBOLS)


def tokenize_smiles(smiles) -> str:
    """
    将 SMILES 规范化并按 SMI_SYMBOLS 分词后再拼回字符串。
    内部自带安全处理，避免 None / NaN 报错。
    """
    # None / NaN 直接返回空字符串
    if smiles is None:
        return ""
    if isinstance(smiles, float) and math.isnan(smiles):
        return ""

    # 其他类型统一转字符串
    if not isinstance(smiles, str):
        smiles = str(smiles)

    smiles = SPACE_NORMALIZER.sub("", smiles)
    tokens = SMI_REGEX.findall(smiles)
    return "".join(tokens)


# =========================
# decode_label_from_binary
# =========================
def decode_label_from_binary(df_atoms_single_smiles, binary_label):
    """
    输入:
        df_atoms_single_smiles: 单个 SMILES 字符串
        binary_label: 长度 = len(tokenize_smiles(smiles)) 的 0/1 列表
    输出:
        positions: 在“仅保留原子字符”列表中的下标列表，例如 [0,1,3]
    """
    # 先得到规范化后的字符串
    filename_without_extension_re = tokenize_smiles(df_atoms_single_smiles)
    indexed_chars = [(i, char) for i, char in enumerate(filename_without_extension_re)]

    # 记录所有“原子字符”在原始字符串里的位置
    atom_positions = [i for i, char in indexed_chars if char.isalpha()]

    # 原始位置 -> 在 atom_positions 里的索引 (反查)
    pos2atom_index = {pos: idx for idx, pos in enumerate(atom_positions)}

    # 找出二值向量中为 1 的位置
    one_indices = [i for i, v in enumerate(binary_label) if v == 1]

    # 映射到第几个原子字符
    positions = []
    for idx in one_indices:
        if idx in pos2atom_index:
            positions.append(pos2atom_index[idx])

    return positions


# =========================
# 安全解析 predicted_atoms 列
# =========================
def safe_eval_pred(x):
    """
    将单元格里的 predicted_atoms 解析成 0/1 列表：
    - NaN / 空值 -> 返回空列表 []
    - 字符串形式的列表，如 "[0, 1, 0]" -> 解析为 [0,1,0]
    - 已经是 list/tuple -> 直接转成 list[int]
    - 其他解析失败 -> 返回空列表 []
    """
    # 已经是 list / tuple
    if isinstance(x, (list, tuple)):
        return [int(i) for i in x]

    # pandas 的 NaN / None
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # 解析失败，按空列表处理；如果想严格一点可以改成 raise
        return []

    if isinstance(v, (list, tuple)):
        return [int(i) for i in v]
    else:
        # 比如 "1" 之类的，转成单元素列表
        try:
            return [int(v)]
        except Exception:
            return []


# =========================
# 读取 Excel & 执行 decode
# =========================
def process_excel(input_file, output_file):
    df = pd.read_excel(input_file)

    if SMILES_COL not in df.columns:
        raise KeyError(f"找不到 SMILES 列: {SMILES_COL}")
    if PRED_COL not in df.columns:
        raise KeyError(f"找不到预测结果列: {PRED_COL}")

    atom_index_list = []

    for i, row in df.iterrows():
        smiles = row[SMILES_COL]
        pred_raw = row[PRED_COL]

        # 统一处理 smiles
        # None / NaN -> 空字符串；其余都转成 str
        if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
            smiles_clean = ""
        else:
            smiles_clean = str(smiles)

        # 安全解析 predicted_atoms
        pred_list = safe_eval_pred(pred_raw)

        # 调用 decode 函数
        atom_index = decode_label_from_binary(smiles_clean, pred_list)
        atom_index_list.append(atom_index)

    # 新增一列存结果
    df[OUT_COL] = atom_index_list

    df.to_excel(output_file, index=False)
    print(f"已保存到: {output_file}")


if __name__ == "__main__":

    head = '7_7_50k'
    attn = 'del'
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    network = "transformer"
    batch_size = 1
    epochs = 50
    learn_rate = 5e-05
    dropout = 0.3
    weight_decay = 0
    best_val_mcc = 0.72
    save_name = "network"
    path_excel_atoms_input = (f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_'
                              f'{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_mcc_{best_val_mcc}_atoms.xlsx')
    result_file_name = f'data/result/statistics_supervision/uspto_yang/shield/ruizhen_product/{head}/result/{save_name}'
    result_folder = os.path.join(parent_dir, result_file_name)
    input_file = os.path.join(result_folder, path_excel_atoms_input)
    path_excel_atoms_output = (f'network_{network}_batch_{batch_size}_epochs_{epochs}_learn_rate_{learn_rate}_drop_'
                               f'{dropout}_weight_delay_{weight_decay}_head_{head}_attn_{attn}_best_val_mcc_{best_val_mcc}_atoms_index.xlsx')
    output_file=os.path.join(result_folder, path_excel_atoms_output)
    process_excel(input_file, output_file)

