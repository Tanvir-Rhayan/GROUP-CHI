import random

# Step 1: Generate simple data
random.seed(0)
X = [2 * random.random() for _ in range(100)]
y = [4 + 3 * x + random.gauss(0, 1) for x in X]   # true model: y = 4 + 3x + noise

# Add bias (x0 = 1)
X_b = [[1, x] for x in X]   # shape (100, 2)

# Absolute Error Function
def absolute_error(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Step 2: Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate, n_iterations):
    m = len(y)
    theta = [random.random(), random.random()]   # [theta0, theta1]

    for _ in range(n_iterations):
        grad0, grad1 = 0, 0
        for i in range(m):
            prediction = theta[0] * X[i][0] + theta[1] * X[i][1]
            error = prediction - y[i]
            grad0 += error * X[i][0]
            grad1 += error * X[i][1]
        theta[0] -= (2/m) * learning_rate * grad0
        theta[1] -= (2/m) * learning_rate * grad1

    y_pred = [theta[0] * X[i][0] + theta[1] * X[i][1] for i in range(m)]
    mae = absolute_error(y, y_pred)
    return theta, mae

# Step 3: Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, learning_rate, n_epochs):
    m = len(y)
    theta = [random.random(), random.random()]   # [theta0, theta1]

    for _ in range(n_epochs):
        for _ in range(m):
            i = random.randrange(m)
            x0, x1 = X[i]
            yi = y[i]
            pred = theta[0] * x0 + theta[1] * x1
            err = pred - yi
            theta[0] -= 2 * learning_rate * err * x0
            theta[1] -= 2 * learning_rate * err * x1

    y_pred = [theta[0] * x0 + theta[1] * x1 for x0, x1 in X]
    mae = absolute_error(y, y_pred)
    return theta, mae

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
