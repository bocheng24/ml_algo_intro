import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression():

    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))

        return g
    
    def predict(self, x, w, b):
        
        m = x.shape[0]
        g_wb = np.zeros(m)

        Z = np.dot(x, w) + b
        G = np.array([self.sigmoid(z)for z in Z ])

        for i in range(m):
            if G[i] > 0.5:
                g_wb[i] = 1

        return g_wb, G
    
    def compute_cost(self, x, y, w, b):

        m = x.shape[0]

        _, G = self.predict(x, w, b)

        total_cost = (-y) * np.log(G) - (1 - y) * np.log(1 - G)
        cost = total_cost.mean()

        return cost
    
    def compute_gradient(self, x, y, w, b):

        m, n = x.shape

        _, f_wb = self.predict(x, w, b)

        errs = f_wb - y
        
        djdw = np.zeros((m, n))

        for i in range(m):
            djdw[i] = errs[i] * x[i]
        
        djdw = djdw.mean(axis = 0)
        djdb = errs.mean()

        return djdw, djdb
    
    def gradient_descent(self, x, y, w_in, b_in, alpha, num_iters):

        J_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):

            djdw, djdb = self.compute_gradient(x, y, w, b)

            w -= alpha * djdw
            b -= alpha * djdb

            if i < 100000:
                J_history.append(self.compute_cost(x, y, w, b))
            
            if i % math.ceil(num_iters / 10) == 0:
                print(f'Iteration: {i:4d}: Cost: {J_history[-1]}    ')
            
        return w, b, J_history

    

def main():

    pass

if __name__ == '__main__':
    main()

    sum([4.99249409e-01, 9.97527377e-01, 1.49389479e+00, -6.10280934e-04, -3.34028437e-05, -2.75356911e-05])