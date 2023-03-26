import numpy as np

class Linear_Regression():
    
    def compute_cost(self, x, y, w, b):

        f_wb = self.predict(x, w, b)
        m = y.shape[0]
        
        total_cost = np.sum((y - f_wb) ** 2) / (2 * m)

        return total_cost

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
    w = 100
    b = 100

    cost = lr.compute_cost(x, y, w, b)
    print(cost)

if __name__ == '__main__':
    main()