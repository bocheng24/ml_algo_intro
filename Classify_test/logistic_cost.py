import numpy as np
from Classification.logistic_regression import Logistic_Regression

X_train = np.array(
    [
        [0.5, 1.5], 
        [1, 1], 
        [1.5, 0.5], 
        [3, 0.5], 
        [2, 2], 
        [1, 2.5]
    ]
)

y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([1, 1])
b_tmp = -3

lgr = Logistic_Regression()

cost = lgr.compute_cost(X_train, y_train, w_tmp, b_tmp)

print(cost)

w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print('Cost for b = -3', lgr.compute_cost(X_train, y_train, w_array1, b_1))
print('Cost for b = -4', lgr.compute_cost(X_train, y_train, w_array2, b_2))