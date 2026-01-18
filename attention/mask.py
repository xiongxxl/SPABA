import numpy as np


def mask_matrix():
    mask_matrix = np.ones((8, 8))
    # kill attention heads
    # mask_matrix[0, :] = 0
    # mask_matrix[4, 0:5] = 0
    # mask_matrix[4, 6:8] = 0
    # mask_matrix[4, 6:8] = 0
    mask_matrix[7, 0:7] = 0
    # print('mask_matrix:',mask_matrix)
    return mask_matrix