import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

W = {"mean": np.mean,
     "median": np.median,
     "std": np.std}


def build_original_sample(scale, size):
    sample = np.random.normal(loc=0, scale=scale, size=size)
    return sample


def simulate_sample_means(estimate, number_of_samples):
    sample_means = []
    for sample_n in range(number_of_samples):
        sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
        sample_mean = estimate(sample)
        sample_means.append(sample_mean)
    return sample_means


def build_simulated_distribution(sample):
    plt.figure(figsize=(8, 6))
    sns.histplot(sample, bins=50, kde=True, color='#0d3b66', edgecolor='#faf0ca', alpha=0.6)
    plt.title("Sample Distribution", fontsize=18, weight='bold')
    plt.xlabel("Sample values", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


original_sample = build_original_sample(10, 1000)
simulated_sample_means = simulate_sample_means(W["mean"], number_of_samples=10000)

build_simulated_distribution(original_sample)
build_simulated_distribution(simulated_sample_means)


def build_confidence_interval():
    return 0
