import numpy as np
import pandas as pd
import math
import glob
from tqdm import tqdm
from decimal import Decimal
import sys

def find_values(file_path):
    #reads the csv value
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
            print(f'Error with filepath {file_path}', repr(e))
    #placing the buy column into Y so it is the one hot encoded vector
    Y = data['buy']
    data.drop(columns = 'buy',axis = 1, inplace=True)
    #data.drop('Unnamed: 0', axis=1, inplace=True)
    #placing the stock values into X
    X = data
    X = X.to_numpy()
    Y = Y.to_numpy()

    return X,Y


def add_w_and_b(input,w,b):
    #adds the weights and biases for the given input 
    output = np.dot(input,w.T)+b
    return output
def loss(input, y):
    #calculates the binary cross entrpy loss
    samples = len(y)
    clipped_input = np.clip(input,1e-7,1-1e-7)
    correct_confidences = clipped_input[range(samples),y]
    negative_log_likelihood = -np.log(correct_confidences)
    return negative_log_likelihood
def mean_loss(loss):
    #returns the mean of the loss
    mean = np.mean(loss)
    return mean
def relu(input):
    #rectified linear unit activation function
    output = np.maximum(0,input)
    return output
def relu_deriv(hidden):
    #derivative of relu function
    hidden = np.where(hidden > 0, 1, 0)
    return hidden
def weight_deriv(input):
    #the derivative of the weights is just the input but i put this here so i dont get confused later
    return input
def bias_deriv(bias):
    #derivative of the bias
    output = np.ones(bias.shape)
    return output
def softmax(output_layer):
    #softmax activation function for the final layer
    exp_layer = np.exp(output_layer - np.max(output_layer,axis=1,keepdims=True))
    norm_values = exp_layer/ np.sum(exp_layer, axis =1,keepdims=True)
    return norm_values

def loss_deriv(input,y):
    #derivative of loss
    result = [[0, 1] if x == 1 else [1, 0] for x in y]
    result = np.array(result)
    deriv = input - result
    return deriv

z5 = [[1.0,2.0],[0.5,1]]

   
hidden5 = softmax(z5)
ccentropy_loss = loss(hidden5,[1,0])


lossDz5 =  loss_deriv(hidden5,[1,0])
#print(hidden5)
#print(ccentropy_loss)
print(lossDz5)

    
