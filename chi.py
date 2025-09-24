import numpy as np

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)   # true model: y = 4 + 3x + noise

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]   # shape (100, 2)

# Absolute Error Function
def absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Step 2: Linear Regression with Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.1, n_iter=1000, method="batch"):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.method = method
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.random.rand(n, 1)

        if self.method == "batch":
            for _ in range(self.n_iter):
                gradients = (2/m) * X.T.dot(X.dot(self.theta) - y)
                self.theta -= self.lr * gradients

        elif self.method == "stochastic":
            for _ in range(self.n_iter):
                for i in range(m):
                    idx = np.random.randint(m)
                    xi, yi = X[idx:idx+1], y[idx:idx+1]
                    gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                    self.theta -= self.lr * gradients

        return self

    def predict(self, X):
        return X.dot(self.theta)

    def mae(self, X, y):
        return absolute_error(y, self.predict(X))

# Step 3: User Input
lr = float(input("Enter learning rate: "))
iterations = int(input("Enter number of iterations for Batch GD: "))
epochs = int(input("Enter number of epochs for SGD: "))

# Step 4: Run Batch GD
bgd_model = LinearRegressionGD(learning_rate=lr, n_iter=iterations, method="batch").fit(X_b, y)
theta_bgd, mae_bgd = bgd_model.theta.ravel(), bgd_model.mae(X_b, y)

# Step 5: Run SGD
sgd_model = LinearRegressionGD(learning_rate=lr, n_iter=epochs, method="stochastic").fit(X_b, y)
theta_sgd, mae_sgd = sgd_model.theta.ravel(), sgd_model.mae(X_b, y)

# Step 6: Results
print("\nBatch Gradient Descent:")
print("Parameters (theta):", theta_bgd)
print("Absolute Error:", mae_bgd)

print("\nStochastic Gradient Descent:")
print("Parameters (theta):", theta_sgd)
print("Absolute Error:", mae_sgd)

# Step 7: Compare performance
if mae_bgd < mae_sgd:
    print("\nBatch Gradient Descent is better for this dataset.")
elif mae_sgd < mae_bgd:
    print("\nStochastic Gradient Descent is better for this dataset.")
else:
    print("\nBoth methods perform equally well on this dataset.")
