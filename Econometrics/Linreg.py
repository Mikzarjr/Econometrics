from Econometrics.Tools import *


class LinearRegression:
    def __init__(self, X_values: list = None, Y_values: list = None, MF: str = None):
        self.X_values = np.array(X_values) if X_values is not None else self.LinRegInput("X")
        self.Y_values = np.array(Y_values) if Y_values is not None else self.LinRegInput("Y")
        self.ModelFunction = self.LinRegModelFunctionInput(MF)

        self.X = self.CalculateXMatrix()
        self.Y = self.CalculateYMatrix()

        self.betas = self.CalculateBetas()
        self.hat_matrix = self.CalculateHatMatrix()

        self.Y_pred = self.CalculatePred()

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

    def CalculatePred(self) -> np.ndarray:
        Y_pred = multiply_matrices(self.X, self.betas)
        return Y_pred

    def CalculateMiddle(self) -> np.ndarray:
        Xt = transpose_matrix(self.X)
        XtX = multiply_matrices(Xt, self.X)
        inverse = inverse_matrix(XtX)
        middle = multiply_matrices(inverse, Xt)
        return middle

    def CalculateHatMatrix(self) -> np.ndarray:
        middle = self.CalculateMiddle()
        hat_matrix = multiply_matrices(self.X, middle)
        return hat_matrix

    def CalculateBetas(self) -> np.ndarray:
        middle = self.CalculateMiddle()
        betas = multiply_matrices(middle, self.Y)
        return betas

    def CalculateXMatrix(self) -> np.ndarray:
        X = []
        for x in self.X_values:
            row = []
            for term in self.ModelFunction:
                a = eval(term)
                row.append(a)
            X.append(row)

        X = np.array(X)
        # print('X:', X)
        return X

    def CalculateYMatrix(self) -> np.ndarray:
        Y = transpose_matrix(self.Y_values)
        # print('Y:', Y)
        return Y

    def CalculateSST(self) -> float:
        SST = 0
        y_avg = np.mean(self.Y_values)
        for y in self.Y_values:
            SST += (y - y_avg) ** 2

        return SST

    def CalculateSSR(self) -> float:
        SSR = 0
        for i in range(len(self.Y_values)):
            SSR += (self.Y_values[i] - self.Y_pred[i]) ** 2

        return SSR

    def CalculateSSE(self) -> float:
        SSE = 0
        y_avg = np.mean(self.Y_values)
        for y in self.Y_pred:
            SSE += (y_avg - y ** 2)

        return SSE


R = LinearRegression([5, 10, 4],
                     [2, 1, 3],
                     "1B + xB")
R.CalculateXMatrix()
R.CalculateBetas()

print(f'Betas: {R.betas}')
print(f'Y predicted: {R.Y_pred}')
print(R.CalculateSST())
print(R.CalculateSSR())
print(R.CalculateSSE())
