import numpy as np

class Linear_Regression():
    

    def predict(self, x, w, b):

        m = x.shape[0]
        f_wb = np.zeros(m)
        
        for i in range(m):
            f_wb[i] = w * x[i] + b

        return f_wb

    def __repr__(self):
        return 'Linear Regression Model'

def main():
    lr = Linear_Regression()
    x = np.array([1, 2])
    y = np.array([300, 500])
    w = 200
    b = 100

    f_wb = lr.predict(x, w, b)
    print(f_wb)

if __name__ == '__main__':

    main()