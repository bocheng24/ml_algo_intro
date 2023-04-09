import numpy as np

class Feature_Scaling:

    def z_score(self, X):

        mu = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)

        X_Norm = (X - mu) / std

        return X_Norm

def main():

    X = np.random.rand(10, 3) * 2000 

    