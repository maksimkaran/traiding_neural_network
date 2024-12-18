import numpy as np
import pandas as pd
import math
import glob
from tqdm import tqdm
from decimal import Decimal
import sys
def find_values(file_path):
   
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
            print(f'Error with filepath {file_path}', repr(e))
    Y = data['buy']
    data.drop(columns = 'buy',axis = 1, inplace=True)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    X = data
    X = X.to_numpy()
    #print(X)
    #sys.exit()
    '''X_norm = []
    for i in range(1,data.shape[1]):
        if (X[:,i].max()-X[:,i].min()) != 0:
            column = ((2*(X[:,i]-X[:,i].min()))/(X[:,i].max()-X[:,i].min()))-1
        else:
            column = np.ones(len(X[:,i]))
            print(column,"AAAAAAAA")
       
        X_norm.append(column)
    X_norm = np.array(X_norm)'''
    Y = Y.to_numpy()

    return X,Y

def standardise_input(input):
    std = np.sum((input-input.mean())**2)/len(input)
    std = np.sqrt(std)
    output = ((input-input.mean()))/std
    return output
def add_w_and_b(input,w,b):
    output = np.dot(input,w.T)+b
    return output
def loss(input, y):
    samples = len(y)
    clipped_input = np.clip(input,1e-7,1-1e-7)
    correct_confidences = clipped_input[range(samples),y]
    #print(np.sum(correct_confidences > 0.5)/len(y))
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

'''def softmax_deriv(input,y):
    softmax_out = np.empty((input.shape[0], input.shape[1]))
    helper = np.empty((input.shape[1],))
    for i in range(input.shape[0]):
        if y[i] == 1:
            helper[0] = input[i][0]*(1-input[i][1])
            helper[1] = -input[i][0]*input[i][1]
        else:
            helper[1] = input[i][0]*(1-input[i][1])
            helper[0] = -input[i][0]*input[i][1]
        softmax_out[i] = helper
    return softmax_out'''



  

def loss_deriv(input,y):# MISILIM DA JE OVDE PROBLEM TAKO DA SLEDECI PUT KAD RADIS OVO PROMENI
    result = np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(len(y))])
    deriv = input - result
    return deriv

filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/train_data/*.csv')]
step_size = 0.0001
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
update_frequency = 1
for i in tqdm(range(100)):
    counter = 0
    w1=w2=w3=b1=b2=b3 = 0
    accuracy = 0
    for file_path in filepaths:#tqdm ubaci posle
        
        x,y = find_values(file_path)
        counter +=1
        #print('///////',x,y,'///////')
        z1 = add_w_and_b(x,weights1,biases1)
        hidden1 = relu(z1)
        z2 = add_w_and_b(hidden1,weights2,biases2)
        hidden2 = relu(z2)

        z3 = add_w_and_b(hidden2,weights3,biases3)
        hidden3 = softmax(z3)
        ccentropy_loss = loss(hidden3,y)
        mean_ccentropy_loss = mean_loss(ccentropy_loss)
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
