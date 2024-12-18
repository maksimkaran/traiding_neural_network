import numpy as np
def standardise_input(input):
    std = np.sum((input-input.mean())**2)/len(input)
    std = np.sqrt(std)
    output = ((input-input.mean()))/std
    return output
def add_w_and_b(input,w,b):
    output = np.dot(input,np.array(w).T)+b
    return output
def loss(input, y):
    samples = len(y)
    clipped_input = np.clip(input,1e-7,1-1e-7)
    correct_confidences = clipped_input[range(samples),y]
    negative_log_likelihood = -np.log(correct_confidences)
    return negative_log_likelihood
def mean_loss(loss):
    mean = np.mean(loss)
    return mean
def relu(input):
    output = np.maximum(0,input)
    return output
def relu_deriv(hidden):
    hidden = np.where(hidden > 0, 1, 0)
    return hidden
def weight_deriv(input):
    return input
def bias_deriv(bias):
    output = np.ones(bias.shape)
    return output
def softmax(output_layer):
    exp_layer = np.exp(output_layer - np.max(output_layer,axis=1,keepdims=True))
    norm_values = exp_layer/ np.sum(exp_layer, axis =1,keepdims=True)
    return norm_values

def softmax_deriv(softmax_output,y):
       # Number of examples
    y = y.reshape(-1, 1)
    n = softmax_output.shape[0]
    
    # Initialize Jacobian matrix with zeros
    jacobian = np.zeros((n, 2, 2))
    jacobian_edited = np.zeros((n, 1, 2))
    for i in range(n):
        s = softmax_output[i].reshape(-1, 1)  # Reshape to (2, 1)
        jacobian[i] = np.diagflat(s.flatten()) - np.dot(s, s.T)  # Correctly calculate the Jacobian
        jacobian_edited[i] = jacobian[i][y[i]]
    jacobian_edited = jacobian_edited.squeeze(axis=1)
    return jacobian_edited

def loss_deriv(input,y):# MISILIM DA JE OVDE PROBLEM TAKO DA SLEDECI PUT KAD RADIS OVO PROMENI
    samples = len(y)
    clipped_input = np.clip(input,1e-7,1-1e-7)
    
    correct_confidences = clipped_input[range(samples),y]
    deriv = -1/(correct_confidences)
    return deriv
input1 = [[0.3,1.1,-0.2],
          [-0.4,-0.3,0.2],
          [0.3,0.1,-1.1],
          [-0.3,-0.1,0.9],
          [0.2,0.4,-0.1]]
correct_indicator = [0,1,1,0,0]
weights1 = [[0.4,-0.7,0.49],
            [0.49,-0.78,0.2],
            [-0.2,0.08,-0.6]]

weights2 = [[0.5,-0.2,0.29],
            [-0.39,0.68,0.8]]

biases1 = [[0.3,-0.1,0.4]]
biases2 = [[0.1,-0.2]]
input1 = np.array(input1)
weights1 = np.array(weights1)
weights2 = np.array(weights2)
biases1 = np.array(biases1)
biases2 = np.array(biases2)
input1 = standardise_input(input1)
correct_indicator = np.array(correct_indicator)

#np.set_printoptions(precision=2)
delta = 0.1

for i in range(40):
    
    z1 = add_w_and_b(input1,weights1,biases1)
    hidden1 = relu(z1)
    
    z2 = add_w_and_b(hidden1,weights2,biases2)
    hidden2 = softmax(z2)
    
    ccentropy_loss = loss(hidden2,correct_indicator)
    loss_derivative_helper = loss_deriv(hidden2,correct_indicator).reshape((len(ccentropy_loss),1))

    hidden1_deriv = relu_deriv(hidden1)

    lossDhidden2 = softmax_deriv(hidden2,correct_indicator)
    lossDweights2 = np.dot(lossDhidden2.T,weight_deriv(hidden1))
    lossDbiases2 = np.sum(lossDhidden2,axis=0)
    lossDbiases2 = lossDbiases2.reshape((len(lossDbiases2), 1))

    lossDhidden1 = np.dot(lossDhidden2,weight_deriv(weights2))
    lossDz1 = np.multiply(lossDhidden1,hidden1_deriv)
    lossDweights1 = np.dot(lossDz1.T,weight_deriv(input1))
    lossDbiases1 = np.sum(lossDz1,axis=0)
    lossDbiases1 = lossDbiases1.reshape((len(lossDbiases1), 1))
    

    mean_ccentropy_loss = mean_loss(ccentropy_loss)
    print(mean_ccentropy_loss)
    weights1 =  weights1+ 0.01*lossDweights1
    weights2 =  weights2+ 0.01*lossDweights2
    biases1 = biases1 + 0.01*lossDbiases1.T
    biases2 = biases2 + 0.01*lossDbiases2.T #ne znam zasto ali ovde sam morao da stavim plus posto se povecavao loss iz nekog razloga
    
    


   
    




