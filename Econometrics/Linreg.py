import re

import sympy as sp

from Econometrics.Tools import *


class LinearRegression:
    def __init__(self, X_values=None, Y_values=None, MF=None):
        self.X_values = X_values if X_values is not None else self.LinRegInput("X")
        self.Y_values = Y_values if Y_values is not None else self.LinRegInput("Y")
        self.ModelFunction = MF if MF is not None else self.LinRegModelFunctionInput()

        self.X = None
        self.Y = transpose_matrix(self.Y_values)
        self.Y_pred = None
        self.betas = None
        self.hat_matrix = None

    def CalculatePred(self):
        if not self.betas:
            self.CalculateBetas()
        self.Y_pred = multiply_matrices(self.X, self.betas)

    def CalculateMiddle(self):
        Xt = transpose_matrix(self.X)
        XtX = multiply_matrices(Xt, self.X)
        inverse = inverse_matrix(XtX)
        middle = multiply_matrices(inverse, Xt)

        return middle

    def CalculateHatMatrix(self):
        middle = self.CalculateMiddle
        hat_matrix = multiply_matrices(self.X, middle)

        self.hat_matrix = hat_matrix

    def CalculateBetas(self):
        middle = self.CalculateMiddle()
        self.betas = multiply_matrices(middle, self.Y)

    def show_true_pred(self):
        Y_pred = multiply_matrices(self.X, self.betas)
        # self.Y_pred = Y_pred
        for i in range(len(self.Y)):
            print(f"Y_real: {self.Y[i]}, Y_pred: {Y_pred[i]}")

    @staticmethod
    def LinRegInput(name="matrix"):
        print(f"Input values of {name}:")
        matrix = list(map(int, input().split()))
        return matrix

    def LinRegModelFunctionInput(self):
        print(f"Input Linear Regression function as 'y = aB0 + bB1 + cB2..'"
              f"Where a, b, c.. are polynomials of x\n"
              f"(Use common notation like 2*x^4B0 + ...")

        def split_formula(formula):
            formula = formula.replace(' ', '')
            terms = re.findall(r'([+-]?\d*x?\^?\d*)B\d+', formula)
            return terms

        def evaluate_symbolic_expression(formula, x_value):
            x = sp.Symbol('x')
            expression = sp.sympify(formula)
            result = expression.subs(x, x_value)
            return result

        formula = input()
        terms = split_formula(formula)
        X = []
        for x in self.X_values:
            rows = []
            for term in terms:
                result = evaluate_symbolic_expression(term, x)
                rows.append(result)
            X.append(rows)

        self.X = transpose_matrix(X)
        print(self.X)

    def find_betas(self):
        X_t = transpose_matrix(self.X)
        one = multiply_matrices(X_t, self.X)
        print("X_t * X:")
        show_matrix(one)
        two = inverse_matrix(one)
        print("Inverse:")
        show_matrix(two)
        three = multiply_matrices(X_t, self.Y)
        print("X_t * Y:")
        show_matrix(three)
        self.betas = multiply_matrices(two, three)
        return self.betas

    def run_algo(self):
        print("Running algorithm...")
        betas = self.find_betas()
        print("\nBetas:")
        show_matrix(betas)
        print("\nY_real, Y_pred:")
        self.show_true_pred()

    def CalcSST(self):
        return 0


R = LinearRegression()
R.run_algo()
