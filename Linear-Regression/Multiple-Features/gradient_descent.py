'''
    Gradient descent method implementation for Multiple-Feature Linear Regression. In this particular case, the model was tested with the classic housing price prediction, but now considering four features instead of just the size.
'''

import numpy as np
import matplotlib.pyplot as plt

# reduce the # of decimal digits diplayed at print
np.set_printoptions(precision = 2)

''' Training set: 

    Size    # bedrooms   # floors   Age   Price(1000s dollars)
    2104          5         1        45       460
    1416          3         2        40       232
    852           2         1        35       178

'''

# X is a matrix; every row of training set consists of a vector, so each row of the matrix represents one example.

# since the target (y) is a single scalar, all targets are grouped into a single output vector
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


# ** the prediction is a scalar **

def make_prediction(x, w, b):
    # x is a vector with n features.
    # w is a vector with n weights
    # b is a scalar.
    # f_wb = w.x + b
    f_wb = np.dot(x,w) + b
    return f_wb

# x test is the 2nd row of X_train, contains [1416, 3, 2, 40]
# the correct target for x_test is already known (y_train[1]): 460

def cost_function(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost += (f_wb - y[i])**2
    return cost/(2*m)

# calculate del_J/del_w and del_J/_del_db

def calculate_gradient(X, y, w, b):
    # each w_j has a partial derivative. For every row (i) will be computed a partial derivative (n in total because X is mxn matrix.)
    m, n = X.shape
    
    # dj_dw will be a vector 
    # dj_db will be a scalar
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    
    # for each row
    for i in range(0, m):
        
        dif = (np.dot(X[i], w) + b) - y[i]
        dj_db = dj_db + dif
    
        for j in range (0, n):
            dj_dw[j] = dj_dw[j] + dif*X[i][j]
    
    return dj_dw/m, dj_db/m

def gradient_descent(X, y, w_init, b_init, alpha, n_iter):
    w = w_init
    b = b_init

    for i in range(0, n_iter):
        # to update the parameters it's necessary compute the gradient
        dj_dw, dj_db = calculate_gradient(X, y, w, b)
        ### scalar*numpy_array = mult. the scalar to each term of the array
        #   in this case, all w_j terms are multiplied by alpha.
        w = w - (alpha*dj_dw)
        b = b - (alpha*dj_db)
    return w, b


# starting at b = 0 and w = [0, 0, 0, 0]

w_init = np.zeros(X_train.shape[1])
b_init = 0.0
n_iter = 1000
alpha = 5.0e-7

w_found, b_found = gradient_descent(X_train, y_train, w_init, b_init, alpha, n_iter)

print(f"The model found that these are the optimized w's and b for this data set:\nW: {w_found}; B: {b_found}")
print("\nPredicting the known values to test the model's accuracy:")

for i in range(X_train.shape[0]):
    f_wb = np.dot(X_train[i], w_found) + b_found
    error = -100 + (f_wb/y_train[i])*100
    print(f"Prediction: {f_wb:.2f} ~ Target: {y_train[i]} - Error: {error:0.2f}%")
