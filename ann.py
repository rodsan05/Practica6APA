import numpy as np

def sigmoid(X):
	return 1/(1 + np.exp(-X))

def sigmoid_prime(X):
	return (1-X) * X

#########################################################################
# NN
#

def predict(thetas, X):
    """
	Predict the label of an input given a trained neural network.

	Parameters
	----------
	thetas : array_like
		Weights for every layer in the neural network.
        It has len = L.

	X : array_like
		The image inputs having shape (number of examples x image dimensions).

	Return 
    ------
	activations : array_like
		Predictions vector containing the activation of each layer.
		It has a length equal to the number of layers.
	"""

    m = X.shape[0]
    activations = []
    al = np.hstack([np.ones((m, 1)), X])
    activations.append(al)


    for i in range(len(thetas) - 1):
        al = forward_propagation(thetas[i], al)
        m = al.shape[0]
        al = np.hstack([np.ones((m, 1)), al])
        activations.append(al)

    aL = forward_propagation(thetas[len(thetas) - 1], al) 
    activations.append(aL)

    return activations

def forward_propagation(theta, a):
	return sigmoid(np.dot(a, theta.T))

def cost(thetas, X, y, aL, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    thetas : array_like
		Weights for every layer in the neural network.
        It has len = L.

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    m = X.shape[0]

    J = (-1/m) * np.sum(y*np.log(aL) + (1-y)*np.log(1-aL))

    regularized_thetas = 0
    for i in range(len(thetas)):
        theta_fix = thetas[i][:,1:]
        regularized_thetas += np.sum(np.power(theta_fix, 2))
    
    J += (lambda_ / (2 * m)) * regularized_thetas

    return J

def cost_predict(thetas, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    thetas : array_like
		Weights for every layer in the neural network.
        It has len = L.

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    m = X.shape[0]

    activations = predict(thetas, X)
    aL = activations[len(activations) - 1]

    J = (-1/m) * np.sum(y*np.log(aL) + (1-y)*np.log(1-aL))

    regularized_thetas = 0
    for i in range(len(thetas)):
        theta_fix = thetas[i][:,1:]
        regularized_thetas += np.sum(np.power(theta_fix, 2))
    
    J += (lambda_ / (2 * m)) * regularized_thetas
    
    return J

def backprop(thetas, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    thetas : array_like
		Weights for every layer in the neural network.
        It has len = L.

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grads : array_like
        Gradient of the cost function with respect to weights
        for each layer in the neural network.
        It has len = L.

    """

    activations = predict(thetas, X)

    L = len(thetas)

    deltas = [None] * L
    deltaL = (activations[L] - y)
    deltas[L-1] = deltaL

    i = len(thetas) - 2
    while (i >= 0):
        delta = np.dot(deltas[i+1], thetas[i+1]) * sigmoid_prime(activations[i+1])
        deltas[i] = delta[:,1:] #esto hace que se ajusten bien los tamaños
        i -= 1

    m = y.shape[0]

    grads = []
    i = len(thetas) - 2
    for i in range(len(thetas)):
        grad = (1/m) * np.dot(deltas[i].T, activations[i])
        grads.append(grad)

    J = cost(thetas, X, y, activations[len(thetas)], lambda_)

    for i in range(L):
        grads[i][:,1:] += (lambda_ / m) * thetas[i][:,1:]

    return J, grads

def gradient_descent(X, y, alpha, epochs, thetas, lambda_):
    history = []
    for epoch in range(epochs):
        J, gradients = backprop(thetas, X, y, lambda_)
        
        for i in range(len(thetas)):
             thetas[i] -= alpha * gradients[i]
        
        history.append(J)

    return history, thetas
