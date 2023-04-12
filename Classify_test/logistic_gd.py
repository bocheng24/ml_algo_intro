import numpy as np
from Classification.logistic_regression import Logistic_Regression
from sklearn.linear_model import LogisticRegression

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
alph = 0.05
iters = 50000

w_out, b_out, _ = lgr.gradient_descent(X_tmp, y_tmp, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

print('\nScikit Learn Logistic Regression --------------------')

skl_lgr = LogisticRegression()
skl_lgr.fit(X_tmp, y_tmp)
w_sk = skl_lgr.coef_[0]
b_sk = skl_lgr.intercept_[0]

print(f'Scikit Learn w: {w_sk}')
print(f'Scikit Learn b: {b_sk}')


y_pred = skl_lgr.predict(X_tmp)
print(f'Scikit Learn predicts: {y_pred}')

f_wb = lgr.predict(X_tmp, w_out, b_out)
f_wb_sk = lgr.predict(X_tmp, w_sk, b_sk)
print(f'Logistic Model predicts: {f_wb}')
print(f'Logistic Model predicts with sklearn: {f_wb_sk}')
# print(skl_lgr.score(X_tmp, y_tmp))