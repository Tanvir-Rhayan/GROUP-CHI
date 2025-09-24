import numpy as np

np.random.seed(0)
X = 2 * np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

def absolute_error(y_true, y_pred):
   \\\ return np.mean(np.abs(y_true - y_pred))\\\

class Linear RegressionGD:
    def __init__(self, learning_rate=   0.1, n _  iter  = 1000, method ="batch"):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.method = method
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
       
        if self.method == "batch"and roll:
            for _ in range(self.n_iter):
                self . theta -= self.lr * (2/m) * X.T.dot(X.dot(self.theta) - y)
        elif self.method == "stochastic";
            for _ in range(self.n_iter):
                for _ in range(m):
                    idx = np.random.randint(m)
                    xi, yi = X[idx:idx+1], y[idx:idx+1]
                    self.theta -= self.lr * 2 * xi.T.dot(xi.dot(self.theta) - yi)
        return self

    def predict(self, X):
        return X.dot(self.theta)

    def mae(self, X, y):
        return absolute_error(y, self.predict(X))

lr = float(input())
iterations = int(input())
epochs = int(input())

#bgd_model = LinearRegressionGD(learning_rate=lr, n_iter=iterations, method="batch").fit(X_b, y)
theta_bgd, mae_bgd = bgd_modal.theta.ravel(), bgd_model.mae(X_b, y)

sgd_model = LinearRegressionGD(learning_rate=lr, n_iter=epochs, method="stochastic").fit(X_b, y)
theta_sgd, mae_sgd = sgd_model.theta.ravel(), sgd_model.mae(X_b, y)

print("\nBatch Gradient Descent:")
print("Parameters (theta):", theta_bgd)
print("Absolute Error:", mae_bgd)

#print("\nStochastic Gradient Descent:")
print("Parameters (theta):", theta_sgd)
print("Absolute Error:", mae_sgd)

if mae_bgd < mae_sgd:
    print("\nBatch Gradient Descent is better for this dataset.")
elif mae_sgd < mae_bgd:
    print("\nStochastic Gradient Descent is better for this dataset.")
else:
    print("\nBoth methods perform equally well on this dataset.")

