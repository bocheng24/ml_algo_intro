import math
import numpy as np

class Linear_Regression():
    
    def compute_cost(self, x, y, w, b):

        f_wb = self.predict(x, w, b)
        m = y.shape[0]
        
        total_cost = np.sum((y - f_wb) ** 2) / (2 * m)

        return total_cost

    def compute_gradient(self, x, y, w, b):

        f_wb = self.predict(x, w, b)

        djdw = np.mean((f_wb - y) * x)
        djdb = np.mean(f_wb - y)

        # m = x.shape[0]
        # djdw = 0
        # djdb = 0

        # for i in range(m):

        #     f_wb_i = w * x[i] + b

        #     djdw += (f_wb_i - y[i]) * x[i]
        #     djdb += f_wb_i - y[i]

        # djdw = djdw / m
        # djdb = djdb / m

        return djdw, djdb
    
    def gradient_descent(self, x, y, w_in, b_in, alpha, num_iters):

        J_history = []
        p_history = []

        w = w_in
        b = b_in

        for i in range(num_iters):
            djdw, djdb = self.compute_gradient(x, y, w, b)
            w -= alpha * djdw
            b -= alpha * djdb

            if i < 100000:
                cost = self.compute_cost(x, y, w, b)
                J_history.append(cost)
                p_history.append([w, b])

            if i % math.ceil(num_iters / 10) == 0:
                output = f'''Iteration {i : 4}: Cost {J_history[-1]:0.2e} djdw: {djdw : 0.3e}, djdb: {djdb : 0.3e} w: {w : 0.3e}, b: {b : 0.5e}'''
                print(output)
        
        return w, b, J_history, p_history

    def predict(self, x, w, b):
        '''
        Generalize predict function:

        x - m, n matrix
        w - n, 1 matrix
        b - sigular number
        '''
        
        f_wb = np.dot(x, w) + b
        
        return f_wb

    def __repr__(self):
        return 'Linear Regression Model'

def main():
    
    x_train = np.random.rand(3, 2)
    w = np.random.rand(2)
    b = 1

    lr = Linear_Regression()
    f_wb = lr.predict(x_train, w, b)

    print(f_wb)

if __name__ == '__main__':
    main()