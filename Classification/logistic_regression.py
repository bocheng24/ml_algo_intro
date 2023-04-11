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
    

def main():

    pass

if __name__ == '__main__':
    main()