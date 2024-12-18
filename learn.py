import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import glob
from tqdm import tqdm
from finta import TA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

def get_candle_data():
    filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/*.csv')]
    for p in tqdm(filepaths):
        try:
            candle_data = pd.read_csv(p)
        except Exception as e:
            print(f'Error with filepath {p}', repr(e))
        
        #candle_data = candle_data.drop(columns = 'datetime',axis = 1)

        candle_data['RSI'] = TA.RSI(candle_data,14)
        candle_data['ATR'] = TA.ATR(candle_data,14)

        candle_data['upper_gap'] = 1.2 * candle_data.ATR 
        candle_data['lower_gap'] = 1.6 * candle_data.ATR
        bbands = TA.BBANDS(candle_data,30,std_multiplier=2)
        candle_data = candle_data.join(bbands)
        bb_bandwith =(candle_data.BB_UPPER-candle_data.BB_LOWER)/candle_data.BB_MIDDLE
        candle_data['Bandwith'] = bb_bandwith
        candle_data =candle_data.dropna()  
        candle_data['buy'] = 0
        candle_data = candle_data.astype({'buy':int})
        symbol = p.split('\\')[-1].split('.')[0]
        candle_data.to_csv(f'D:/bruh/trading_deep_learning/{symbol}.csv')

def get_buy_and_train():
    filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/*.csv')]
    i = 0
 
    global X
    X = pd.DataFrame()
    global Y
    Y = pd.DataFrame()
    prices = pd.DataFrame()
    for p in tqdm(filepaths):
        try:
            candle_data = pd.read_csv(p)
        except Exception as e:
            print(f'Error with filepath {p}', repr(e))
        if candle_data.empty == True:
            
            continue
        for i in candle_data.index:
            try:
                long_entry_condition = candle_data.shift(-5).close+(candle_data.open*0.02) > candle_data.open
            except:
                print("final trades")
        candle_data.loc[long_entry_condition,'buy'] = 1
        bought = candle_data[candle_data.buy == 1]
        not_bought = candle_data[candle_data.buy == 0]
        if (len(not_bought) == 0) or (len(bought) == 0):
                continue
        if len(bought)>len(not_bought):
            bought = bought.sample(n=len(bought)-len(not_bought))
        else:
            not_bought = bought.sample(n=len(not_bought)-len(bought))
        candle_data = pd.concat([bought,not_bought], axis = 0)
        Y = pd.concat([candle_data,Y])
        candle_data.drop(columns = 'buy',axis = 1, inplace=True)
        candle_data.drop(columns = 'datetime',axis = 1, inplace=True)
        X = pd.concat([candle_data,X])
        prices = pd.concat([X,prices])
    Y  = Y['buy']
    scaler = StandardScaler()
    scaler.fit(X)
    standardised_data = scaler.transform(X)
    X = standardised_data 
def derivative_RELU(input):
    if input<0:
        input = 0
    else:
        input = 1
    return input


get_buy_and_train()
E = math.e

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

class Activation_RELU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values/ np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
    def backward(self,input):
        #ova funkcija vraca 2d np listu koja sadrzi izvod softmax funkcije koju posle mozes da koristis za mnozenje u chain rule
        n_features = input.shape[1]
  
        # Initialize the Jacobian
        
        self.result = []
        # Vectorized computation
        for i in range(input.shape[0]):
            jacobian = np.zeros((n_features, n_features))
            for n in range(n_features):
                for z in range(n_features):
                    if n==z:
                        jacobian[n,z] = input[i,n] * (1- input[i,z])
                    else:
                        jacobian[n,z] = -input[i,n] * input[i,z] 
            derivs = np.sum(jacobian, axis=0)
            self.result.append(derivs)
            
        self.result = np.array(self.result)
       





class Loss:
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        derivs = self.backward(output,y)
        print(derivs)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        print(y_pred.shape)
        print("y shape",y_true.shape)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true] #uzima vrednost tacne klase npr: tacno-[0,1] predvidjeno-[0.2,0.8] pravi listu sa clanom 0.8 
            #print(correct_confidences)
        elif len(y_true.shape) ==2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self,y_pred,y_true):
        deriv = np.zeros((293,0))
        print(y_true)
        for i in range(len(y_true)):
            try:
                if y_true.iloc[0,i] == 1:
                    deriv[i] = -(1/y_pred[i,1])
                else:
                    deriv[i] = 0
            except:
                print("deleted index")
        print(deriv)    
layer1 = Layer_Dense(17,5)
activation1 = Activation_RELU()
layer2 = Layer_Dense(5,2)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
predicitons = np.argmax(activation2.output,axis=1)
activation2.backward(activation2.output)
accuracy = np.mean(predicitons == Y)
loss_funciton = Loss_CategoricalCrossentropy()
loss = loss_funciton.calculate(activation2.output,Y)
print("accuracy:",accuracy)
print("loss:",loss)

#print(activation2.output)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=1)




