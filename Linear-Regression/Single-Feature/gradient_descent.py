'''
    Gradient descent method implementation for a single-feature Linear Regression model: houses' price prediction

    since gradient descent needs a cost function to find the minima and for each small step towards it need to compute the partial derivatives of J wrt w and b, there are three separated functions to compute all needed values.

'''

import numpy as np
import matplotlib.pyplot as plt

# dataset
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

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

def plot_cost_iteration(cost_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    ax1.plot(cost_hist[:100])
    ax1.set_title("Cost x iteration (first 100)")
    ax1.set_ylabel("Cost"); ax2.set_ylabel("Cost")
    ax1.set_xlabel('iteration step'); ax2.set_xlabel('iteration step')
    ax2.plot(1000 + np.arange(len(cost_hist[1000:])), cost_hist[1000:])
    ax2.set_title("Cost x iteration (end)")

    plt.show()

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
    cost_over_time = []
    parameters_over_time = []

    # save all changes in j and w, b over time to compare, but up to 10.000 values


    for i in range(n_iter):
        dj_dw, dj_db = calculate_gradient(x, y, w, b)
        # updating both parameters 
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        if i<10000:
            cost_over_time.append(cost_function(x, y, w, b))
            parameters_over_time.append([w,b])

    return w, b, cost_over_time, parameters_over_time

def make_prediction(w_found, b_found, feature):
    return w_found*feature + b_found

w_init = 0
b_init = 0
number_of_iterations = 10000
alpha = 1.0e-2

w_found, b_found, cost_over_time, wb_over_time = gradient_descent(w_init, b_init, alpha, number_of_iterations, x_train, y_train, cost_function, calculate_gradient)

value_to_predict = float(input("House's size in sqft to predict: "))/1000

predicted_value = make_prediction(w_found, b_found, value_to_predict)

print(f"w found: {w_found}\nb found: {b_found}")
print(f"Price prediction for {value_to_predict*1000} sqft: {predicted_value:0.2f} thousand dollars.")

plot_cost_iteration(cost_over_time)
