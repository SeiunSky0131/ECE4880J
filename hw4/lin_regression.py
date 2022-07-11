import numpy as np

# Template code for gradient descent, to make the implementation
# a bit more straightforward.
# Please fill in the missing pieces, and then feel free to
# call it as needed in the functions below.
def gradient_descent(X, y, lr, num_iters):
    losses = []
    n, d = X.shape
    w = np.zeros((d,1))
    for i in range(num_iters):
        grad = -2*X.T@y + 2*X.T@X@w
        w = w - lr * grad
        loss = np.sqrt((X@w - y)**2)
        losses.append(loss)
    return losses, w

# Code for (2-5)
def linear_regression():
    X = np.array([[1,1],[2,3],[3,3]])
    Y = np.array([[1],[3],[3]])    

    ##### ADD YOUR CODE FOR ALL PARTS HERE
    # 2-3. direct calculation
    # LAMBDA = 1 # change the number here
    # my_lambda = np.array([[LAMBDA,0],[0,LAMBDA]])
    # w = np.linalg.inv(X.T@X + my_lambda)@X.T@Y
    # print(w)

    # 4-5. Gradient decent
    losses, w = gradient_descent(X,Y,0.01, 10000)
    print(w)

if __name__ == "__main__":
    linear_regression()
    


