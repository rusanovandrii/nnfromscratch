import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
import os
import pickle

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the arrays
train_images = train_images.reshape(60000, -1).T / 255.0
test_images = test_images.reshape(10000, -1).T / 255.0

# Initialize random parameters
def initialize_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# ReLU
def ReLU(Z1):
    return np.maximum(Z1, 0)

# Softmax
def softmax(x):
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU
def ReLU_deriv(Z):
    return Z > 0

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Loss function
def cross_entropy_loss(Y_true, Y_pred):
    m = Y_true.shape[1]
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    loss = -1 / m * np.sum(Y_true * np.log(Y_pred))
    return loss   

# Save parameters to a file
def save_params(W1, b1, W2, b2):
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    with open("params.pkl", "wb") as f:
        pickle.dump(params, f)

# Load parameters from a file
def load_params():
    with open("params.pkl", "rb") as f:
        params = pickle.load(f)
    return params["W1"], params["b1"], params["W2"], params["b2"]


# Train the model with gradient descent
def gradient_descent(X, Y, alpha, iterations):
    # Check if parameters file exists
    if os.path.exists("params.pkl"):
        # Load parameters
        print("Loading parameters")
        W1, b1, W2, b2 = load_params()
        
    else:
        # Initialize parameters
        W1, b1, W2, b2 = initialize_params()
        loss_history = []
        
        for i in range(iterations):
            # Forward propagation
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            
            # Calculate loss
            loss = cross_entropy_loss(one_hot(Y), A2)
            loss_history.append(loss)
            
            # Backward propagation
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            
            # Update parameters
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        # Plot Loss
        plt.plot(loss_history)
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
    
    # Save parameters and return them
    save_params(W1, b1, W2, b2)
    return W1, b1, W2, b2  
    
alpha = 0.1
iterations = 500
W1, b1, W2, b2  = gradient_descent(train_images, train_labels, alpha, iterations)

# Get a prediction for a single digit
def get_prediction(image, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, image)
    prediction = np.argmax(A2, axis=0)
    return prediction



# Generate a random index
random_index = np.random.randint(train_images.shape[1])

# Select a random image from the training set
image = train_images[:, random_index]
test = image
test = test.reshape(-1, 1)
print(test.shape)

# Reshape the image back to 28x28
image_reshaped = image.reshape(28, 28)

# Plot the image
plt.imshow(image_reshaped, cmap='gray')
plt.show()

print(get_prediction(test, W1, b1, W2, b2))
