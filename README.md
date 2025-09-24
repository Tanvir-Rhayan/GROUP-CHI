# Linear Regression using Batch Gradient Descent and Stochastic Gradient Descent

**Group Number:** χ  

**Group Members:**  
- Mst. Maris Islam - 2024000000012  
- Tanvir Rahyan Shayem - 2024000000035  
- Md. Monjurul Haque Moni - 2024000000038  
- Saykat Kumar Modak - 2023200000380  
- Niaj Mahamud - 2023200000289  
- Md. Golam Saroare Shuvo - 2022000000045  

**Submitted To:**  
[TMD] Tashreef Muhammad, Lecturer, Dept. of CSE, Southeast University, Bangladesh  

---

## Project Topic

Implement and explain **Stochastic Gradient Descent (SGD)** and **Batch Gradient Descent (BGD)** for Linear Regression. Compare convergence speed and performance on a dataset (synthetic or real).

This project demonstrates the implementation of **Linear Regression** using two types of gradient descent:

- **Batch Gradient Descent (BGD):** Updates parameters using all training samples at once.  
- **Stochastic Gradient Descent (SGD):** Updates parameters one sample at a time, converging faster but with small fluctuations.

---

## Objective

- Understand the difference between **BGD** and **SGD** for optimizing linear regression.  
- Compare **convergence speed** and **accuracy** for both methods.  
- Implement both methods in **pure Python** (no external libraries required).  
- Calculate and analyze the **Mean Absolute Error (MAE)** for model evaluation.

---

## About the Code

The Python code (`linear_regression_gd.py`) is designed to be **beginner-friendly**. Key features include:

1. **Synthetic Dataset:** Generates 100 random samples `(x, y)` following a linear relationship `y = 4 + 3x + noise`.  
2. **Parameter Initialization:** θ0 and θ1 are initialized randomly.  
3. **Batch Gradient Descent:**  
   - Updates parameters using **all samples per iteration**.  
   - Computes **Mean Absolute Error (MAE)** after training.  
4. **Stochastic Gradient Descent:**  
   - Updates parameters **one random sample at a time** per epoch.  
   - Computes **Mean Absolute Error (MAE)** after training.  
5. **Comparison:** Compares the **MAE** of BGD and SGD and prints which method performs better on the dataset.

---

## Key Functions

- `absolute_error(y_true, y_pred)`: Computes **Mean Absolute Error (MAE)**.  
- `batch_gradient_descent(X, y, learning_rate, n_iterations)`: Implements **Batch Gradient Descent**.  
- `stochastic_gradient_descent(X, y, learning_rate, n_epochs)`: Implements **Stochastic Gradient Descent**.  

---

## How to Run

1. Clone the repository:  
```bash
git clone <your-repo-url>
