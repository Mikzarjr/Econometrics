import numpy as np


def transpose_matrix(matrix):
    tr_mat = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return tr_mat


def show_matrix(matrix):
    for row in matrix:
        print(row)


def inverse_matrix(matrix):
    return np.linalg.inv(matrix)


def multiply_matrices(matrix_1, matrix_2):
    return np.dot(matrix_1, matrix_2)
