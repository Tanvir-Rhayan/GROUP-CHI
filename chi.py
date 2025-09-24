import numpy as np

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)   # shape (100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)   # true model: y = 4 + 3x + noise

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]   # shape (100, 2)

# Absolute Error Function
def absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Step 2: Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)
    theta = np.random.rand(X.shape[1], 1)  # [theta0, theta1]

    for _ in range(n_iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients

    y_pred = X.dot(theta)
    mae = absolute_error(y, y_pred)
    return theta.ravel(), mae

# Step 3: Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, learning_rate=0.1, n_epochs=50):
    m = len(y)
    theta = np.random.rand(X.shape[1], 1)

    for _ in range(n_epochs):
        for i in range(m):
            idx = np.random.randint(m)
            xi = X[idx:idx+1]
            yi = y[idx:idx+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients

    y_pred = X.dot(theta)
    mae = absolute_error(y, y_pred)
    return theta.ravel(), mae

# Step 4: User Input
lr = float(input("Enter learning rate: "))
iterations = int(input("Enter number of iterations for Batch GD: "))
epochs = int(input("Enter number of epochs for SGD: "))

# Step 5: Run and Compare
theta_bgd, mae_bgd = batch_gradient_descent(X_b, y, lr, iterations)
theta_sgd, mae_sgd = stochastic_gradient_descent(X_b, y, lr, epochs)

print("\nBatch Gradient Descent:")
print("Parameters (theta):", theta_bgd)
print("Absolute Error:", mae_bgd)

print("\nStochastic Gradient Descent:")
print("Parameters (theta):", theta_sgd)
print("Absolute Error:", mae_sgd)

# Step 6: Compare performance
if mae_bgd < mae_sgd:
    print("\nBatch Gradient Descent is better for this dataset.")
elif mae_sgd < mae_bgd:
    print("\nStochastic Gradient Descent is better for this dataset.")
else:
    print("\nBoth methods perform equally well on this dataset.")
