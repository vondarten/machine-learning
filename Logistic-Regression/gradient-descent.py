import numpy as np
import matplotlib.pyplot as plt

# Gradient descent method implementation for Logistic Regression Model

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def calculate_gradient(X, y, w, b):
    
    # m = #of examples
    # n = #of of variables (features)
    m, n = X.shape

    # for each row there are n derivatives since there's one derivative per variable (each variable changes in a different way)
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    # for each example (...)
    for i in range(m):
        # z is the linear regression prediction (y^)
        z = np.dot(X[i],w) + b
        # f_wb is sigmoid of "z"
        f_wb = sigmoid(z)
        # compare the value predicted to the correct traning answer
        diff = f_wb - y[i]

        # the gradient equation needs the sum over the difference times each individual x_j 
                
        for j in range(n):
            derivative = diff*X[i][j]
            dj_dw[j] += derivative
        
        dj_db += diff

    return dj_dw/m, dj_db/m

def gradient_descent(X, y, w_init, b_init, alpha, n_iter):
    w = w_init
    b = b_init

    for i in range(n_iter):
        dj_dw, dj_db = calculate_gradient(X, y, w, b)
        w -= alpha*dj_dw
        b -= alpha*dj_db

    return w, b

w_i = np.zeros_like(X_train[0])
b_i = 0.0
alpha = 0.1
n_iters = 100000

w_found, b_found = gradient_descent(X_train, y_train, w_i, b_i, alpha, n_iters)

print(f"w found: {w_found}")
print(f"b found: {b_found}")