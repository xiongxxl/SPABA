import re

# 正则表达式
SPACE_NORMALIZER = re.compile(r"\s+")
SMI_SYMBOLS = r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|" + \
             r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"

# 合法原子符号
ATOM_SYMBOLS = {
    'Li','Be','Na','Mg','Al','Si','Cl','Ca','Zn','As','Se','se','Br','Rb','Sr','Ag',
    'Sn','Te','te','Cs','Ba','Bi','H','B','C','N','O','F','P','S','K','I',
    'b','c','n','o','p','s'
}

def tokenize_smiles(line):
    return re.findall(SMI_SYMBOLS, line.strip())

def atom_flags_padded(smiles, max_len=512):
    tokens = tokenize_smiles(smiles)
    flags = [1 if token in ATOM_SYMBOLS else 0 for token in tokens]
    # 补零到 max_len
    if len(flags) < max_len:
        flags += [0] * (max_len - len(flags))
    else:
        flags = flags[:max_len]  # 若过长则截断
    return flags

if __name__=='__main__':
    # 示例
    smiles = "CCOC(CBr)CC=C"
    result = atom_flags_padded(smiles)
    print(result)
    print(f"Length: {len(result)}")  # 应该是 512