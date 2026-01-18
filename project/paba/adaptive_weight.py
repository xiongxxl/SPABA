import numpy as np


def find_and_keep_largest_block(matrix):
    matrix=matrix.tolist()
    if not matrix or not matrix[0]:
        return []

    rows, cols = len(matrix), len(matrix[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def dfs(x, y):
        stack = [(x, y)]
        cells = []
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < rows and 0 <= cy < cols and not visited[cx][cy] and matrix[cx][cy] == 1:
                visited[cx][cy] = True
                cells.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))
        return cells

    max_block = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                current_block = dfs(i, j)
                if len(current_block) > len(max_block):
                    max_block = current_block

    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for x, y in max_block:
        result[x][y] = 1

    return result
#
# if __name__ == "__main__":
#
#     # 示例用法
#     matrix = [
#                 [1, 0, 0, 1, 1],
#                 [1, 1, 0, 0, 0],
#                 [0, 0, 1, 0, 0],
#                 [0, 1, 1, 1, 0],
#                 [0, 0, 0, 0, 1]
#               ]
#
#     matrix_a = np.load('4_0.npy')
#     result = find_and_keep_largest_block(matrix_a)
#     for row in result:
#         print(row)
