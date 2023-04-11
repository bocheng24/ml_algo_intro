import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression():

    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))

        return g
    

def main():

    z = np.arange(-10, 11)
    lgr = Logistic_Regression()

    y = lgr.sigmoid(z)

    df = pd.DataFrame({'z': z, 'y': y})
    print(df)

    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    ax.plot(z, y)

    plt.show()

if __name__ == '__main__':
    main()