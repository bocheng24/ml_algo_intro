import numpy as np
from Classification.logistic_regression import Logistic_Regression

x = np.array(
    [
        [2, 3], 
        [4, 5], 
        [6, 7]
    ]
)

y = np.array([0, 1, 0]).reshape(-1, 1)

w = np.array([-1, -2])
b = 0.5

lgr = Logistic_Regression()

g_wb, g = lgr.predict(x, w, b)

print(g_wb)
print(g)