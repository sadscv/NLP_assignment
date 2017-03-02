import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    if len(data.shape) >= 2:
        N = data.shape[0]
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(h1, W2) + b2
    h2 = softmax(z2)
    cost = - np.sum(np.log(h2[labels == 1])) / N
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    dz2 = np.zeros_like(z2)
    dz1 = np.zeros_like(z1)
    grad_W2 = np.zeros_like(W2)
    grad_W1 = np.zeros_like(W1)
    grad_b1 = np.zeros_like(b1)
    grad_b2 = np.zeros_like(b2)


    dz2 = (h2 - labels) / N
    dz1 = np.dot(dz2, W2.T) * sigmoid_grad(h1)
    grad_W2 = np.dot(h1.T, dz2)
    grad_b2 = np.sum(dz2, axis=0, keepdims=True)
    grad_W1 = np.dot(data.T, dz1)
    grad_b1 = np.sum(dz1, axis=0, keepdims=True)


    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((grad_W1.flatten(), grad_b1.flatten(),
        grad_W2.flatten(), grad_b2.flatten()))
    
    return cost, grad






def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
                                                          dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()