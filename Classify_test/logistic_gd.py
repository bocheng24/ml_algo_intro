import numpy as np
from Classification.logistic_regression import Logistic_Regression

X_tmp = np.array(
    [
        [0.5, 1.5], 
        [1, 1], 
        [1.5, 0.5], 
        [3, 0.5], 
        [2, 2], 
        [1, 2.5]
    ]
)

y_tmp = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.array([2.,3.])

b_tmp = 1.

lgr = Logistic_Regression()
dj_db_tmp, dj_dw_tmp = lgr.compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp)

print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

w_tmp  = np.zeros_like(X_tmp[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = lgr.gradient_descent(X_tmp, y_tmp, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")