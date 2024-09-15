import numpy as np


class Regression:
    def __init__(self, X=None, Y=None):
        self.X = X if X is not None else self.input_matrix("X")
        self.Y = Y if Y is not None else self.input_matrix("Y")
        self.betas = None
        self.Y_pred = None

    @staticmethod
    def transpose_matrix(matrix):
        tr_mat = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
        return tr_mat

    @staticmethod
    def show_matrix(matrix):
        for row in matrix:
            print(row)

    def show_true_pred(self):
        Y_pred = self.multiply_matrices(self.X, self.betas)
        # self.Y_pred = Y_pred
        for i in range(len(self.Y)):
            print(f"Y_real: {self.Y[i]}, Y_pred: {Y_pred[i]}")

    @staticmethod
    def input_matrix(name="matrix"):
        print(f"Input {name} row by row (empty line to finish):")
        matrix = []
        while True:
            row = input()
            if row.strip() == "":
                break
            row = list(map(float, row.split()))
            matrix.append(row)
        return np.array(matrix)

    @staticmethod
    def inverse_matrix(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def multiply_matrices(matrix_1, matrix_2):
        return np.dot(matrix_1, matrix_2)

    def find_betas(self):
        X_t = self.transpose_matrix(self.X)
        one = self.multiply_matrices(X_t, self.X)
        print("X_t * X:")
        self.show_matrix(one)
        two = self.inverse_matrix(one)
        print("Inverse:")
        self.show_matrix(two)
        three = self.multiply_matrices(X_t, self.Y)
        print("X_t * Y:")
        self.show_matrix(three)
        self.betas = self.multiply_matrices(two, three)
        return self.betas

    def run_algo(self):
        print("Running algorithm...")
        betas = self.find_betas()
        print("\nBetas:")
        self.show_matrix(betas)
        print("\nY_real, Y_pred:")
        self.show_true_pred()

    # def CalcSST(self):


R = Regression()
R.run_algo()
