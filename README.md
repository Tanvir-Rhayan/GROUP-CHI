# GROUP_CHI
Topic-16: Implement and explain SGD and Batch GD for Linear Regression. Compare convergence speed and performance on a dataset (synthetic or real).     
ID: ( 2022000000045 , 2023200000380 , 2024000000012 , 2024000000035 , 2024000000038 )
# Linear Regression using Batch Gradient Descent and Stochastic Gradient Descent

**Group Number:** θ  
**Group Members:**  
- Mst. Maris Islam - 2024000000012  
- Tanvir Rahyan Shayem - 2024000000035  
- M1  
- M2  
- M3  
- M4  

**Submitted To:**  
[TMD] Tashreef Muhammad, Lecturer, Dept. of CSE, Southeast University, Bangladesh  

---

## Project Topic

**Implement and explain Stochastic Gradient Descent (SGD) and Batch Gradient Descent (BGD) for Linear Regression. Compare convergence speed and performance on a dataset (synthetic or real).**

This project demonstrates the implementation of **Linear Regression** using two types of gradient descent:

- **Batch Gradient Descent (BGD):** Updates parameters using all training samples at once.  
- **Stochastic Gradient Descent (SGD):** Updates parameters one sample at a time, converging faster but with small fluctuations.

---

## Objective

- Learn the difference between BGD and SGD for optimizing linear regression.  
- Compare **convergence speed** and **accuracy** for both methods.  
- Implement both methods in **pure Python** (no external libraries required).  
- Calculate and analyze the **Absolute Error (MAE)** for model evaluation.

---

## About the Code

The Python code (`linear_regression_gd.py`) is designed to be **beginner-friendly**. Key features include:

1. **Synthetic Dataset:** Generates 100 random samples `(x, y)` following a linear relationship `y = 4 + 3x + noise`.  
2. **Parameter Initialization:** θ0 and θ1 are initialized randomly.  
3. **Batch Gradient Descent:**  
   - Updates parameters using **all samples per iteration**.  
   - Computes **Absolute Error (MAE)** after training.  
4. **Stochastic Gradient Descent:**  
   - Updates parameters **one random sample at a time** per epoch.  
   - Computes **Absolute Error (MAE)** after training.  
5. **Comparison:** The program compares the **MAE** of BGD and SGD and prints which method performs better on the dataset.

**Key Functions:**

- `absolute_error(y_true, y_pred)`: Computes mean absolute error.  
- `batch_gradient_descent(X, y, learning_rate, n_iterations)`: Implements BGD.  
- `stochastic_gradient_descent(X, y, learning_rate, n_epochs)`: Implements SGD.  

---

## How to Run

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
