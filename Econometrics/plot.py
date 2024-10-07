"""
Add plot function
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set parameters
np.random.seed(42)
B = 10000  # Number of simulations
n = 10  # Number of observations
beta_0 = 2
sigma_u = np.sqrt(4)

# Storage for estimates
beta_0_estimates = []
beta_1_estimates = []
R2_simple = []
R2_polynomial = []

# Simulate data and perform regressions
for _ in range(B):
    # Generate x and y data
    x_i = np.random.normal(0, 1, n).reshape(-1, 1)
    u_i = np.random.normal(0, sigma_u, n)
    y_i = beta_0 + u_i

    # Simple regression y_i = b0 + b1 * x_i
    reg = LinearRegression().fit(x_i, y_i)
    beta_0_estimates.append(reg.intercept_)
    beta_1_estimates.append(reg.coef_[0])

    # Calculate R-squared for simple regression
    R2_simple.append(reg.score(x_i, y_i))

    # Polynomial regression: y_i = b0 + b1 * x_i + b2 * x_i^2 + b3 * x_i^3
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(x_i)
    reg_poly = LinearRegression().fit(X_poly, y_i)

    # Calculate R-squared for polynomial regression
    R2_polynomial.append(reg_poly.score(X_poly, y_i))

# Convert results to numpy arrays for easy handling
beta_0_estimates = np.array(beta_0_estimates)
beta_1_estimates = np.array(beta_1_estimates)
R2_simple = np.array(R2_simple)
R2_polynomial = np.array(R2_polynomial)

# Plot histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Histogram of beta_0 estimates
axs[0, 0].hist(beta_0_estimates, bins=30, color='skyblue', edgecolor='black')
axs[0, 0].axvline(x=beta_0, color='red', linestyle='--', label=r'$\beta_0$')
axs[0, 0].set_title(r"Histogram of $\hat{\beta}_0$")
axs[0, 0].legend()

# Histogram of beta_1 estimates
axs[0, 1].hist(beta_1_estimates, bins=30, color='lightgreen', edgecolor='black')
axs[0, 1].axvline(x=0, color='red', linestyle='--', label=r'$\beta_1$')
axs[0, 1].set_title(r"Histogram of $\hat{\beta}_1$")
axs[0, 1].legend()

# Histogram of R-squared for simple regression
axs[1, 0].hist(R2_simple, bins=30, color='lightcoral', edgecolor='black')
axs[1, 0].set_title(r"Histogram of $R^2$ (Simple)")

# Histogram of R-squared for polynomial regression
axs[1, 1].hist(R2_polynomial, bins=30, color='gold', edgecolor='black')
axs[1, 1].set_title(r"Histogram of $R^2$ (Polynomial)")

plt.tight_layout()
plt.show()
