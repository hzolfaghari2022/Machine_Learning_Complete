# %%
# Part 1: Generate data

import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=50, noise=5.0):
    np.random.seed(42)
    X = np.linspace(-10, 10, n_samples)
    true_slope = 3
    true_intercept = 8
    noise_values = np.random.randn(n_samples) * noise
    y = true_slope * X + true_intercept + noise_values
    return X, y

X, y = generate_data(n_samples=50, noise=10.0)

print("Part 1 completed")

plt.figure()
plt.scatter(X, y)
plt.title("Generated Data")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# %%
# Part 2: Linear Regression

def h_w(x, w):
    return w[0] + w[1] * x

def linear_regression_closed_form(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

w = linear_regression_closed_form(X, y)

print("Part 2 completed")
print(f"w_0, intercept = {w[0]:.2f}")
print(f"w_1, slope     = {w[1]:.2f}")

y_pred = h_w(X, w)

plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Prediction")
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


# %%
# Part 3: Polynomial Regression

def polynomial_features(X, degree):
    X_poly = np.c_[np.ones(len(X))]
    for i in range(1, degree + 1):
        X_poly = np.c_[X_poly, X**i]
    return X_poly

def polynomial_regression(X, y, degree):
    X_poly = polynomial_features(X, degree)
    w = np.linalg.pinv(X_poly).dot(y)
    return w

m = 5
w_poly = polynomial_regression(X, y, m)

print("Part 3 completed")
print(f"Parameters for degree {m}:")
print(w_poly)

X_smooth = np.linspace(X.min(), X.max(), 300)
X_smooth_poly = polynomial_features(X_smooth, m)
y_poly_pred = X_smooth_poly.dot(w_poly)

plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X_smooth, y_poly_pred, label=f"Polynomial Degree {m}")
plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Part 4: Automatically push this project to both GitHub repositories

import subprocess
from datetime import datetime


def run_git_command(command):
    """
    Run a Git command and print the result.
    """
    print(f"\nRunning: {command}")

    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result.returncode


print("\n==============================")
print("Part 4: GitHub Push")
print("==============================")

commit_message = f"Update project files on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Check current Git status
run_git_command("git status")

# Add all modified and new files
run_git_command("git add .")

# Commit changes
commit_code = run_git_command(f'git commit -m "{commit_message}"')

# Push to your GitHub repository
print("\nPushing to your GitHub repository...")
run_git_command("git push origin main")

# Push to your brother's GitHub repository
print("\nPushing to brother GitHub repository...")
run_git_command("git push brother main")

print("\nGitHub push section completed.")