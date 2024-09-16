import numpy as np


def transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix.transpose()


def show_matrix(matrix: np.array) -> None:
    print(matrix)


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


def multiply_matrices(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    return np.dot(matrix_1, matrix_2)
