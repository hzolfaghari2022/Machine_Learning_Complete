import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=50, noise=5.0):
    np.random.seed(42)
    X = np.linspace(-10, 10, n_samples)
    # Ground truth line: y = 3x + 8
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n_samples) * noise
    y = true_slope * X + true_intercept + noise
    return X, y

# plot
X, y = generate_data(n_samples=50, noise=10.0)
plt.scatter(X, y, color='blue', label='Data Points')
plt.title("Generated Data (Univariate)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()