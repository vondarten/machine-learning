'''
    Gradient descent method implementation for a single-feature Linear Regression model: houses' price prediction

    since gradient descent needs a cost function to find the minima and for each small step towards it need to compute the partial derivatives of J wrt w and b, there are three separated functions to compute all needed values.

'''

import numpy as np

# dataset
x_train = np.array([1.0, 2.0, 3.0, 4.5, 5.0])
y_train = np.array([300.0, 500.0, 700.0, 720.0, 900.0])

# cost function
def cost_function(x, y, w, b):
    # nº of training example
    m = len(x)
    cost_sum = 0

    for i in range(0, m):
        f_wb = w*x[i] + b
        cost_sum += (f_wb - y[i])**2
    
    total_cost = cost_sum/(2*m)
    return total_cost

# find del_J/del_w and del_J/del_b
# grad = (dj_dw, dj_db)
# with the gradient its possible to implement the gradient descent algorithm.

def calculate_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(0, m):
        f_wb = w*x[i] + b
        dj_dw += (f_wb - y[i])*x[i]
        dj_db += (f_wb - y[i])
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(w_init, b_init, alpha, n_iter, x, y, cost_function, calculate_gradient):

    w = w_init
    b = b_init

    for i in range(n_iter):
        dj_dw, dj_db = calculate_gradient(x, y, w, b)
        # updating both parameters 
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

    return w, b

def make_prediction(w_found, b_found, feature):
    return w_found*feature + b_found

w_init = 0
b_init = 0
number_of_iterations = 10000
alpha = 1.0e-2

w_found, b_found = gradient_descent(w_init, b_init, alpha, number_of_iterations, x_train, y_train, cost_function, calculate_gradient)

# for example, the price prediction for a 1200 sqft house
value_to_predict = 1.2

predicted_value = make_prediction(w_found, b_found, value_to_predict)

print(f"Price prediction for {value_to_predict*1000} sqft: {predicted_value:0.2f} thousand dollars.")