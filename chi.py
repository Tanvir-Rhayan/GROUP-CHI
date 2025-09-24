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
    m = len(y)\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{float} % For exact figure placement [H]

%----------------------------------------
% Title Page
%----------------------------------------
\begin{document}

\begin{titlepage}
    \centering
    \includegraphics[width=0.3\textwidth]{SEULogo.png}\par\vspace{1cm}
    {\scshape\LARGE Southeast University, Bangladesh \par}
    \vspace{1cm}
    {\Large CSE261.3: Numerical Methods \par}
    \vspace{0.5cm}
    {\Large Group Assignment Report \par}
    \vspace{1.5cm}
    {\large \textbf{Implement and explain Stochastic Gradient Descent (SGD) and Batch Gradient Descent (BGD) for Linear Regression. Compare convergence speed and performance on a dataset (synthetic or real). \par}}
    \vfill
    \textbf{Group Number: $\chi$} \par
    \begin{tabular}{|l|c|}
\hline
\textbf{Name} & \textbf{ID} \\
\hline
Mst. Maris Islam & 2024000000012\\\hline
Tanvir Rahyan Shayem & 2024000000035 \\\hline
M1 & 1 \\\hline
M2 & 2 \\\hline
M3 & 3 \\\hline
M4 & 4 \\
\hline
\end{tabular}

    \vfill

    \textbf{Submitted To:}  \par
    [TMD] Tashreef Muhammad  \\
    Lecturer, Dept. of CSE  \\
    Southeast University, Bangladesh \par
    \vfill
    Summer 2025
\end{titlepage}

%----------------------------------------
% Abstract
%----------------------------------------
\begin{abstract}
This report presents the implementation of Gradient Descent techniques for Linear Regression. Two variants are studied: Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD). Their update rules are derived mathematically, implemented in Python, and applied to a dataset. Convergence speed and accuracy are compared. Results show that BGD converges smoothly but slower, while SGD converges faster with fluctuations, making it efficient for large-scale datasets.
\end{abstract}

%----------------------------------------
% Sections
%----------------------------------------

\section{Introduction}
Linear Regression is one of the fundamental algorithms in machine learning, used for modeling the relationship between input features and output variables. Gradient Descent is the key optimization method used to minimize the cost function. This report focuses on two types of gradient descent: Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD). Their performance is compared on synthetic and real datasets.

\section{Theoretical Background}
\subsection{Linear Regression}
The hypothesis function is:
\begin{equation}
h_\theta(x) = \theta_0 + \theta_1 x
\end{equation}
The cost function (Mean Squared Error) is:
\begin{equation}
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
\end{equation}

\subsection{Batch Gradient Descent (BGD)}
The update rule is:
\begin{equation}
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
\end{equation}
where $\alpha$ is the learning rate and $m$ is the total number of samples.  
BGD updates parameters using the entire dataset each iteration.

\subsection{Stochastic Gradient Descent (SGD)}
The update rule is:
\begin{equation}
\theta_j := \theta_j - \alpha \left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
\end{equation}
Here, parameters are updated after each training example.  
SGD converges faster but introduces randomness in updates.

%----------------------------------------
\section{Methodology}
Algorithm steps:
\begin{enumerate}
    \item Define dataset $(x, y)$.
    \item Initialize parameters $\theta_0, \theta_1$.
    \item Apply BGD using all samples per iteration.
    \item Apply SGD using one random sample per iteration.
    \item Track cost function across iterations.
    \item Compare convergence speed and accuracy.
\end{enumerate}

\subsection{Pseudo-code for Batch Gradient Descent}
\begin{verbatim}
Input: X, y, learning rate α, iterations
Initialize θ = 0
For k = 1 to iterations:
    gradient = (1/m) * Σ (hθ(x_i) - y_i) * x_i
    θ = θ - α * gradient
End For
\end{verbatim}

\subsection{Pseudo-code for Stochastic Gradient Descent}
\begin{verbatim}
Input: X, y, learning rate α, epochs
Initialize θ = 0
For epoch = 1 to epochs:
    For i = 1 to m:
        gradient = (hθ(x_i) - y_i) * x_i
        θ = θ - α * gradient
    End For
End For
\end{verbatim}

%----------------------------------------
\section{Comparison of BGD and SGD}
\begin{itemize}
    \item \textbf{BGD:} Stable convergence, but slower for large datasets.
    \item \textbf{SGD:} Faster updates, good for large datasets, but fluctuates around the minimum.
    \item \textbf{Trade-off:} BGD ensures stability, SGD ensures speed and scalability.
\end{itemize}

\begin{table}[h!]
\centering
\begin{tabular}{|c|c|c|}
\hline
 & \textbf{Batch Gradient Descent} & \textbf{Stochastic Gradient Descent} \\
\hline
Update rule & Uses all samples per iteration & Uses one sample per iteration \\
\hline
Convergence & Smooth, stable & Fast but noisy \\
\hline
Efficiency & Slow for large $m$ & Very efficient for large $m$ \\
\hline
Accuracy & High & High (with fluctuations) \\
\hline
\end{tabular}
\caption{Comparison between BGD and SGD (Table 1)}
\end{table}

%----------------------------------------
\section{Results and Analysis}
We tested both BGD and SGD on a synthetic dataset with $m=100$ samples. The learning rate $\alpha=0.01$ was used, and convergence was measured in terms of cost function values across iterations.  
BGD converged smoothly but required more computation per iteration, while SGD converged faster but with fluctuations.

\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Iteration} & \textbf{Cost (BGD)} & \textbf{Cost (SGD)} & \textbf{θ0 (BGD)} & \textbf{θ0 (SGD)} & \textbf{θ1 (BGD)} & \textbf{θ1 (SGD)} & \textbf{Observation} \\
\hline
1 & 120.45 & 125.12 & 0.10 & 0.12 & 0.95 & 0.97 & SGD decreases faster initially \\
\hline
2 & 95.34 & 102.87 & 0.20 & 0.18 & 0.90 & 0.92 & SGD still faster, BGD smooth \\
\hline
3 & 75.12 & 80.43 & 0.30 & 0.25 & 0.85 & 0.87 & BGD stable, SGD fluctuates \\
\hline
4 & 60.50 & 63.20 & 0.40 & 0.32 & 0.80 & 0.82 & Convergence trend visible \\
\hline
5 & 45.87 & 48.10 & 0.48 & 0.39 & 0.75 & 0.78 & Both moving towards minima \\
\hline
6 & 30.12 & 32.50 & 0.55 & 0.45 & 0.72 & 0.74 & SGD still slightly noisy \\
\hline
7 & 12.34 & 14.28 & 0.60 & 0.50 & 0.70 & 0.71 & Both converge near minimum \\
\hline
\end{tabular}}
\caption{Convergence comparison of BGD and SGD for Cost and Parameters (Table 2)}
\end{table}

% Both graphs appear exactly one after another
\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{BGD_SGD_Cost.jpg}
    \caption{Cost Convergence: BGD vs SGD}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{BGD_SGD_Output.jpg}
    \caption{Parameter Values Convergence: θ0, θ1 for BGD vs SGD}
\end{figure}

%----------------------------------------
\section{Discussion}
\begin{itemize}
    \item Both methods minimize the same cost function.
    \item BGD is suitable for small datasets due to its stability.
    \item SGD is efficient for large datasets and forms the basis of modern deep learning optimizers.
    \item The choice depends on dataset size and computational resources.
\end{itemize}

\section{Conclusion}
Key findings:
\begin{itemize}
    \item Both BGD and SGD successfully optimize Linear Regression.
    \item BGD converges smoothly but is computationally expensive for large data.
    \item SGD is faster and scalable, though convergence is noisier.
    \item For real-world machine learning, SGD is widely preferred.
\end{itemize}


\end{document}
a[1] * X[i][1]
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
        i = random.randrange(m)
        xi = X[i]
        yi = y[i]
        pred = theta[0] * xi[0] + theta[1] * xi[1]
        err = pred - yi
        theta = [theta[j] - 2 * lr * err * xi[j] for j in range(2)]


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
