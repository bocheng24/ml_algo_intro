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
        
        m = x.shape[0]
        f_wb = np.zeros(m)
        
        for i in range(m):
            f_wb[i] = w * x[i] + b
        
        return f_wb

    def __repr__(self):
        return 'Linear Regression Model'

def main():
    lr = Linear_Regression()
    x_train = np.array([1, 2])
    y_train = np.array([300, 500])
    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2

    w_final, b_final, J_hist, p_hist = lr.gradient_descent(x_train, 
                                                       y_train, 
                                                       w_init, 
                                                       b_init, 
                                                       tmp_alpha, 
                                                       iterations
                                                      )
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

if __name__ == '__main__':
    main()