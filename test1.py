import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))  # Subtracting max for numerical stability
    return e_z / e_z.sum(axis=0)

def softmax_derivative(softmax_output):
    # Reshape the softmax output to a column vector if necessary
    s = softmax_output.reshape(-1, 1)
    print(s)
    return np.diagflat(s) - np.dot(s, s.T)

# Example usage:
z = np.array([1.0, 2.0, 3.0])
softmax_output = softmax(z)
print("Softmax Output:", softmax_output)

derivative = softmax_derivative(softmax_output)
print("Softmax Derivative:\n", derivative)