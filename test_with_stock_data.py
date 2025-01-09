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

'''def standardise_input(input):
    #just a standardisation function that returns values which are usualy between -3 and 3
    std = np.sum((input-input.mean())**2)/len(input)
    std = np.sqrt(std)
    output = ((input-input.mean()))/std
    return output'''

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

#finds all the filepaths for all of the .csv files in the training data folder
filepaths = [file for file in glob.glob(f'D:/bruh/trade_copy/traiding_neural_network/train_data/*.csv')]
step_size = 0.00001

#randomly initializing the weights and setting the biases to 0
weights1 = np.random.uniform(low=-0.5, high=0.5, size=(40,15))
weights2 = np.random.uniform(low=-0.5, high=0.5, size=(40,40))
weights3 = np.random.uniform(low=-0.5, high=0.5, size=(40,40))
weights4 = np.random.uniform(low=-0.5, high=0.5, size=(40,40))
weights5 = np.random.uniform(low=-0.5, high=0.5, size=(2,40))
biases1 = np.zeros((1, 40))
biases2 = np.zeros((1, 40))
biases3 = np.zeros((1, 40))
biases4 = np.zeros((1, 40))
biases5 = np.zeros((1, 2))

weights1 = np.array(weights1)
weights2 = np.array(weights2)
weights3 = np.array(weights3)
weights4 = np.array(weights4)
weights5 = np.array(weights5)
biases1 = np.array(biases1)
biases2 = np.array(biases2)
biases3 = np.array(biases3)
biases4 = np.array(biases4)
biases5 = np.array(biases5)
accuracy_avg = 0
update_frequency = 3 #this is freqently the drivatives will be added to the weights and biases
for i in tqdm(range(30000)): #arbitrairy number of itterations 
    counter = 0
    w1=w2=w3=w4=w5=b1=b2=b3=b4=b5 = 0
    accuracy = 0
    helper = 0
    for file_path in filepaths:#for every ticker we find the derivatives
        
        x,y = find_values(file_path)
        
        counter +=1
        z1 = add_w_and_b(x,weights1,biases1)
        hidden1 = relu(z1)
        z2 = add_w_and_b(hidden1,weights2,biases2)
        hidden2 = relu(z2)
        z3 = add_w_and_b(hidden2,weights3,biases3)
        hidden3 = relu(z3)
        z4 = add_w_and_b(hidden3,weights4,biases4)
        hidden4 = relu(z4)
        z5 = add_w_and_b(hidden4,weights5,biases5)
        hidden5 = softmax(z5)
        ccentropy_loss = loss(hidden5,y)
        #print(ccentropy_loss)
        mean_ccentropy_loss = mean_loss(ccentropy_loss)
        #we made the forward step in the lines above and now we are propagating backwards and getting the derivatives
        hidden1_deriv = relu_deriv(hidden1)
        hidden2_deriv = relu_deriv(hidden2)
        hidden3_deriv = relu_deriv(hidden3)
        hidden4_deriv = relu_deriv(hidden4)

        lossDz5 =  loss_deriv(hidden5,y)
        lossDweights5 = np.dot(lossDz5.T,weight_deriv(hidden4))
        lossDbiases5 = np.sum(lossDz5,axis=0)
        lossDbiases5 = lossDbiases5.reshape((len(lossDbiases5), 1))

        lossDhidden4 = np.dot(lossDz5,weight_deriv(weights5))
        lossDz4 = np.multiply(lossDhidden4,hidden4_deriv)
        lossDweights4 = np.dot(lossDz4.T,weight_deriv(hidden3))
        lossDbiases4 = np.sum(lossDz4,axis=0)
        lossDbiases4 = lossDbiases4.reshape((len(lossDbiases4), 1))

        lossDhidden3 = np.dot(lossDz4,weight_deriv(weights4))
        lossDz3 = np.multiply(lossDhidden3,hidden3_deriv)
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
        clipped_input = np.clip(hidden5,1e-7,1-1e-7)
        correct_confidences = clipped_input[range(samples),y]
        #finding the accuracy
        for x in correct_confidences:
            if x > 0.5:
                accuracy += 1
        accuracy_avg += accuracy/len(correct_confidences)
        
        accuracy = 0
        
        
        w1 += lossDweights1
        w2 += lossDweights2
        w3 += lossDweights3
        w4 += lossDweights4
        w5 += lossDweights5
        b1 += lossDbiases1.T
        b2 += lossDbiases2.T
        b3 += lossDbiases3.T
        b4 += lossDbiases4.T
        b5 += lossDbiases5.T

        #updating the weights and biases
        if counter % update_frequency == 0:
            #print(w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)         
              
            weights1 -= (w1/update_frequency)*step_size
            weights2 -= (w2/update_frequency)*step_size
            weights3 -= (w3/update_frequency)*step_size
            weights4 -= (w4/update_frequency)*step_size
            weights5 -= (w5/update_frequency)*step_size
            biases1 -= (b1/update_frequency)*step_size
            biases2 -= (b2/update_frequency)*step_size
            biases3 -= (b3/update_frequency)*step_size
            biases4 -= (b4/update_frequency)*step_size
            biases5 -= (b5/update_frequency)*step_size
            w1=w2=w3=w4=w5=b1=b2=b3=b4=b5 = 0
    
    accuracy_a = accuracy_avg/len(filepaths)
    print(accuracy_a)
    
    accuracy_avg = 0
    
    if i % 10 == 0:
        step_size *= 1
    
    if i% 20 == 0:
        
        #print("\n",accuracy_avg/5)
        
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
    hidden3 = relu(z3)

    z4 = add_w_and_b(hidden3,weights4,biases4)
    hidden4 = relu(z4)

    z5 = add_w_and_b(hidden4,weights5,biases5)
    hidden5 = softmax(z5)

    ccentropy_loss = loss(hidden5,y1)

    mean_ccentropy_loss = mean_loss(ccentropy_loss)
    mean_average_loss +=mean_ccentropy_loss

print(mean_average_loss/len(filepaths1))
