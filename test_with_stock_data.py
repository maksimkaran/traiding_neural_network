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
    data.drop('Unnamed: 0', axis=1, inplace=True)
    #placing the stock values into X
    X = data
    X = X.to_numpy()
    Y = Y.to_numpy()

    return X,Y

def standardise_input(input):
    #just a standardisation function that returns values which are usualy between -3 and 3
    std = np.sum((input-input.mean())**2)/len(input)
    std = np.sqrt(std)
    output = ((input-input.mean()))/std
    return output

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
    result = np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(len(y))])
    deriv = input - result
    return deriv

#finds all the filepaths for all of the .csv files in the training data folder
filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/train_data/*.csv')]
step_size = 0.0001

#randomly initializing the weights and setting the biases to 0
weights1 = np.random.rand(20,14)
weights2 = np.random.rand(10,20)
weights3 = np.random.rand(2,10)
biases1 = np.zeros((1, 20))
biases2 = np.zeros((1, 10))
biases3 = np.zeros((1, 2))

weights1 = np.array(weights1)
weights2 = np.array(weights2)
weights3 = np.array(weights3)
biases1 = np.array(biases1)
biases2 = np.array(biases2)
biases3 = np.array(biases3)

accuracy_avg = 0
update_frequency = 1 #this is freqently the drivatives will be added to the weights and biases
for i in tqdm(range(100)): #arbitrairy number of itterations 
    counter = 0
    w1=w2=w3=b1=b2=b3 = 0
    accuracy = 0
    for file_path in filepaths:#for every ticker we find the derivatives
        
        x,y = find_values(file_path)
        counter +=1
        z1 = add_w_and_b(x,weights1,biases1)
        hidden1 = relu(z1)
        z2 = add_w_and_b(hidden1,weights2,biases2)
        hidden2 = relu(z2)
        z3 = add_w_and_b(hidden2,weights3,biases3)
        hidden3 = softmax(z3)
        ccentropy_loss = loss(hidden3,y)
        mean_ccentropy_loss = mean_loss(ccentropy_loss)
        #we made the forward step in the lines above and now we are propagating backwards and getting the derivatives
        hidden1_deriv = relu_deriv(hidden1)
        hidden2_deriv = relu_deriv(hidden2)
        
        

        lossDz3 =  loss_deriv(hidden3,y)

        lossDweights3 = np.dot(lossDz3.T,weight_deriv(hidden2))
        lossDbiases3 = np.sum(lossDz3,axis=0)
        lossDbiases3 = lossDbiases3.reshape((len(lossDbiases3), 1))

        lossDhidden2 = np.dot(lossDz3,weight_deriv(weights3))
        lossDz2 = np.multiply(lossDhidden2,hidden2_deriv)
        lossDweights2 = np.dot(lossDz2.T,weight_deriv(hidden1))
        lossDbiases2 = np.sum(lossDz2,axis=0)
        lossDbiases2 = lossDbiases2.reshape((len(lossDbiases2), 1))

        lossDhidden1 = np.dot(lossDz2,weight_deriv(weights2))
        lossDz1 = np.multiply(lossDhidden1,hidden1_deriv)
        lossDweights1 = np.dot(lossDz1.T,weight_deriv(x))
        lossDbiases1 = np.sum(lossDz1,axis=0)
        lossDbiases1 = lossDbiases1.reshape((len(lossDbiases1), 1))

        samples = len(y)
        clipped_input = np.clip(hidden3,1e-7,1-1e-7)
        correct_confidences = clipped_input[range(samples),y]
        #finding the accuracy
        for x in correct_confidences:
            if x < 0.5:
                accuracy += 1
        accuracy_avg += accuracy/len(y)
        accuracy = 0
        
        
        w1 += lossDweights1
        w2 += lossDweights2
        w3 += lossDweights3
        b1 += lossDbiases1.T
        b2 += lossDbiases2.T
        b3 += lossDbiases3.T
        #updating the weights and biases
        if counter % update_frequency == 0:
            weights1 -= (w1/update_frequency)*step_size
            weights2 -= (w2/update_frequency)*step_size
            weights3 -= (w3/update_frequency)*step_size
            biases1 -= (b1/update_frequency)*step_size
            biases2 -= (b2/update_frequency)*step_size
            biases3 -= (b3/update_frequency)*step_size
            w1=w2=w3=b1=b2=b3 = 0

    accuracy_avg = accuracy_avg/len(filepaths)
    print("\n",accuracy_avg)
    if i % 10 == 0:
        step_size *= 0.9
    
    if i% 6 == 0:
        
        #print("\n",accuracy_avg/5)
        accuracy_avg = 0
        print(f"sum_loss: {np.sum(ccentropy_loss)}")
        print(f"mean_loss: {mean_ccentropy_loss}\n")
    
    
    
#here we are doing one forward step on the test data to see the difference between the two
mean_average_loss = 0
filepaths1 = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/test_data/*.csv')]
for file_path1 in tqdm(filepaths1):
    x1,y1 = find_values(file_path1)

    z1 = add_w_and_b(x1,weights1,biases1)
    hidden1 = relu(z1)

    z2 = add_w_and_b(hidden1,weights2,biases2)
    hidden2 = relu(z2)

    z3 = add_w_and_b(hidden2,weights3,biases3)
    hidden3 = softmax(z3)

    ccentropy_loss = loss(hidden3,y1)

    mean_ccentropy_loss = mean_loss(ccentropy_loss)
    mean_average_loss +=mean_ccentropy_loss

print(mean_average_loss/len(filepaths1))
