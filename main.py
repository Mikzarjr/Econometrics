from Econometrics.Linreg import LinearRegression

R = LinearRegression(X=[[1, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 0],
                        [1, 1, 1]],
                     Y_values=[1, 2, 3, 4, 5],
                     MF="1B + xB + x^2B")

R.CalculateBetas()

print(f'Betas: {R.betas}')
print(f'Y predicted: {R.Y_pred}')