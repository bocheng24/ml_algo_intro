import math
import numpy as np
import copy
import matplotlib.pyplot as plt

class Linear_Regression():
    
    def compute_cost(self, x, y, w, b):

        f_wb = self.predict(x, w, b)
        m = y.shape[0]
        
        total_cost = np.sum((y - f_wb) ** 2) / (2 * m)

        return total_cost

    def compute_gradient(self, x, y, w, b):

        f_wb = self.predict(x, w, b)

        djdw = 0
        djdb = 0

        for x_i, y_i, f_wb_i in zip(x, y, f_wb):

            djdw += (f_wb_i - y_i) * x_i
            djdb += f_wb_i - y_i

            # print(djdw)

        djdw = djdw / x.shape[0]
        djdb = djdb / x.shape[0]

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
                cost = self.compute_cost(x, y, w, b)
                J_history.append(cost)
               

            if i % math.ceil(num_iters / 10) == 0:
                output = f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   "
                print(output)
        
        return w, b, J_history

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
    
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

    lr = Linear_Regression()
    tmp_dj_dw, tmp_dj_db = lr.compute_gradient(X_train, y_train, w_init, b_init)
    print(f'dj_db at initial w,b: {tmp_dj_db}')
    print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    
    # run gradient descent 
    w_final, b_final, J_hist = lr.gradient_descent(
        X_train, 
        y_train, 
        initial_w, 
        initial_b,
        alpha, 
        iterations
    )

    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

        # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()

if __name__ == '__main__':
    main()