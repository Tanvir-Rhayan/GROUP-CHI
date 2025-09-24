import random

# Step 1: Generate simple data
random.seed(0)
X = [2 * random.random() for _ in range(100)]
y = [4 + 3 * x + random.gauss(0, 1) for x in X]   # true model: y = 4 + 3x + noise

# Add bias term
X_b = [[1, x] for x in X]

# Absolute Error Function
def absolute_error(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Step 2: Batch Gradient Descent
def batch_gradient_descent(X, y, lr, n_iter):
    m = len(y)
    theta = [random.random(), random.random()]

    for _ in range(n_iter):
        grad0 = grad1 = 0
        for i in range(m):
            pred = theta[0] * X[i][0] + theta[1] * X[i][1]
            err = pred - y[i]
            grad0 += err * X[i][0]
            grad1 += err * X[i][1]
        theta[0] -= (2/m) * lr * grad0
        theta[1] -= (2/m) * lr * grad1

    y_pred = [theta[0] * x0 + theta[1] * x1 for x0, x1 in X]
    return theta, absolute_error(y, y_pred)

# Step 3: Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, lr, n_epochs):
    m = len(y)
    theta = [random.random(), random.random()]

    for _ in range(n_epochs):
        for _ in range(m):
            idx = random.randrange(m)
            x0, x1 = X[idx]
            yi = y[idx]
            pred = theta[0] * x0 + theta[1] * x1
            err = pred - yi
            theta[0] -= 2 * lr * err * x0
            theta[1] -= 2 * lr * err * x1

    y_pred = [theta[0] * x0 + theta[1] * x1 for x0, x1 in X]
    return theta, absolute_error(y, y_pred)

# Step 4: User Input
lr = float(input("Enter learning rate: "))
iterations = int(input("Enter number of iterations for Batch GD: "))
epochs = int(input("Enter number of epochs for SGD: "))

# Step 5: Run both methods
theta_bgd, mae_bgd = batch_gradient_descent(X_b, y, lr, iterations)
theta_sgd, mae_sgd = stochastic_gradient_descent(X_b, y, lr, epochs)

# Step 6: Display results
results = [
    ("Batch Gradient Descent", theta_bgd, mae_bgd),
    ("Stochastic Gradient Descent", theta_sgd, mae_sgd)
]

for name, theta, mae in results:
    print(f"\n{name}:")
    print("Parameters (theta):", theta)
    print("Absolute Error:", mae)

# Step 7: Compare performance
if mae_bgd < mae_sgd:
    print("\nBatch Gradient Descent is better for this dataset.")
elif mae_sgd < mae_bgd:
    print("\nStochastic Gradient Descent is better for this dataset.")
else:
    print("\nBoth methods perform equally well on this dataset.")
