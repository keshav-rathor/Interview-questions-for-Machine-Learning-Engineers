import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    # Initialize
    weights = initial_weights.astype(float).copy()
    bias = float(initial_bias)
    n_samples = features.shape[0]
    mse_values = []
    
    for epoch in range(epochs):
        # Forward pass
        z = np.dot(features, weights) + bias
        y_pred = sigmoid(z)
        
        # Loss (MSE)
        mse = np.mean((y_pred - labels) ** 2)
        mse_values.append(round(mse, 4))
        
        # Gradients
        dz = 2 * (y_pred - labels) * y_pred * (1 - y_pred) / n_samples
        dw = np.dot(features.T, dz)       # gradient wrt weights
        db = np.sum(dz)                   # gradient wrt bias
        
        # Update rule
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    # Round for matching expected output
    return list(np.round(weights, 4)), round(bias, 4), mse_values

print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))