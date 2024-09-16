from Econometrics.Tools import *


class LinearRegression:
    def __init__(self, X_values: list = None, Y_values: list = None, MF: str = None):
        self.X_values = np.array(X_values) if X_values is not None else self.LinRegInput("X")
        self.Y_values = np.array(Y_values) if Y_values is not None else self.LinRegInput("Y")
        self.ModelFunction = self.LinRegModelFunctionInput(MF)
        self.X = None
        self.Y = transpose_matrix(self.Y_values)
        self.Y_pred = None
        self.betas = None
        self.hat_matrix = None

    def CalculatePred(self) -> None:
        if not self.betas:
            self.CalculateBetas()
        self.Y_pred = multiply_matrices(self.X, self.betas)

    def CalculateMiddle(self) -> np.ndarray:
        Xt = transpose_matrix(self.X)
        XtX = multiply_matrices(Xt, self.X)
        inverse = inverse_matrix(XtX)
        middle = multiply_matrices(inverse, Xt)

        return middle

    def CalculateHatMatrix(self) -> None:
        middle = self.CalculateMiddle()
        hat_matrix = multiply_matrices(self.X, middle)

        self.hat_matrix = hat_matrix

    def CalculateBetas(self) -> None:
        middle = self.CalculateMiddle()
        self.betas = multiply_matrices(middle, self.Y)

    @staticmethod
    def LinRegInput(name: str = "matrix") -> np.ndarray:
        print(f"Input values of {name}:")
        matrix = np.array(list(map(int, input().split())))
        return matrix

    @staticmethod
    def LinRegModelFunctionInput(MF: str) -> list[str]:

        def get_terms(formula):
            terms = list(formula.split('B'))
            terms.pop(-1)
            return terms

        formula = MF if MF is not None else input(f"Input Linear Regression function as 'y = aB + bB + cB..'"
                                                  f"Where a, b, c.. are polynomials of x\n"
                                                  f"(Use common Python notation like 2B + 3*xB - 2*x**2B + ...\n")
        terms = get_terms(formula)
        return terms

    def CalculateXMatrix(self):
        X = []
        for x in self.X_values:
            row = []
            for term in self.ModelFunction:
                a = eval(term)
                row.append(a)
            X.append(row)

        self.X = transpose_matrix(np.array(X))


R = LinearRegression([1, 2, 3],
                     [1, 2, 3],
                     "2B + 3*xB - 2*3*x**2B")
R.CalculateXMatrix()
R.CalculateBetas()
R.CalculateHatMatrix()

print(R.betas)
