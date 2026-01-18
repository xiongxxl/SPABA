import numpy as np
def remove_outer_layer(square_matrix):
    # 计算新数组的大小
    n = len(square_matrix)
    inner_size = n - 2
    # 创建一个新数组来存储内部元素
    inner_matrix = [[0] * inner_size for _ in range(inner_size)]

    # 复制内部元素到新数组
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            inner_matrix[i - 1][j - 1] = square_matrix[i][j]
            inner_matrix=np.array(inner_matrix)
    return inner_matrix
