
import numpy as np

# 加载 .npy 文件
data = np.load('Brc1ccc2c(c1)CCc1nncn1-2.COC(=O)C1(c2cncc(B(O)O)c2)CC1.npy')

# 查看数组形状（尺寸）
print("数组形状:", data.shape)
print("数组维度:", data.ndim)
print("数组大小（元素总数）:", data.size)
print("数据类型:", data.dtype)

# 查看文件大小（磁盘占用）
import os
file_size = os.path.getsize('your_file.npy')
print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

