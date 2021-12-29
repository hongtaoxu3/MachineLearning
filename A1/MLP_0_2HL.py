import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from random import seed
from random import random
import time
import copy 
# normalize x
def load_dataset(reshape=False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    if reshape:
        X_train = X_train.reshape([X_train.shape[0], -1])
        #X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_test, y_test
X_train, y_train,  X_test, y_test = load_dataset(reshape=True)

#do not normalize x
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#X_train = X_train.reshape([X_train.shape[0], -1])
#X_test = X_test.reshape([X_test.shape[0], -1])

class Layer:
    def __init__(self):
        pass
    def forward(self, input):  
        return input
    def backward(self, input, grad_output):
        num_units = input.shape[1]  
        d_layer_d_input = np.eye(num_units) 
        result=np.dot(grad_output, d_layer_d_input)
        return result

class sigmoid(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out
    def back(self,x):
        return (1.0 - sigmoid(x)) * sigmoid(x)
class tanh(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t
    def back(self ,x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return (1-t*t)
class ReLU(Layer):
    def __init__(self):
        pass   
    def forward(self, input):
        relu_forward = np.maximum(0,input)
        return relu_forward
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class MLP(Layer):
    def __init__(self, input, output, l_rate=0.1):

        self.l_rate = l_rate
        self.weights = np.random.normal(scale = np.sqrt(2/(input+output)),loc=0.0,  
                                        size = (input,output))
        self.bias = np.zeros(output)
        
    def forward(self,input):
        out=np.dot(input,self.weights) + self.bias
        return out
    
    def backward(self,input,grad_output):
        #L2 regularization
        #lambd=0.01
        #grad_input = np.dot(grad_output, self.weights.T)
        #grad_weights = np.dot(input.T, grad_output)+(lambd/input.shape[0])*self.weights
        #grad_bias = grad_output.mean(axis=0)*input.shape[0]
         
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_bias = grad_output.mean(axis=0)*input.shape[0]
        self.weights = self.weights - self.l_rate * grad_weights
        self.bias = self.bias - self.l_rate * grad_bias
        
        return grad_input

network = []
#no hidden layer
#network.append(MLP(X_train.shape[1],784))
#network.append(ReLU())
#network.append(MLP(784,10))

#one hidden layer
#network.append(MLP(X_train.shape[1],784))
#network.append(ReLU())
#network.append(MLP(784,128))
#network.append(ReLU())
#network.append(MLP(128,10))
# 2 hidden layers
network.append(MLP(X_train.shape[1],784))
network.append(ReLU())
network.append(MLP(784,128))
network.append(ReLU())
network.append(MLP(128,128))
network.append(ReLU())
network.append(MLP(128,10))

def crossentropy(logits,Y):
    logp = logits[np.arange(len(logits)),Y]
    entropy = - logp + np.log(np.sum(np.exp(logits),axis=-1))

    #L2 regularization
    #cross_cost=(-1.0)*np.sum(logp)/Y.shape[0]
    #L2=(np.sum(np.square(logp)))*(0.01/(2*Y.shape[0]))
    #entropy = cross_cost+L2
    return entropy
def softmax(logits,Y):
    logp = np.zeros_like(logits)
    logp[np.arange(len(logits)),Y] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return ( -logp + softmax) / logits.shape[0]

def forward(network, X):
    activations = []
    X1=X
    for l in network:
        activations.append(l.forward(X1))
        X1 = activations[-1]
    return activations
def predict(network,X):
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def train(network,X,y):
   
    layer = forward(network,X)
    inputs = [X]+layer
    logits = layer[-1]

    loss = crossentropy(logits,y)
    loss_grad = softmax(logits,y)
    

    for index in range(len(network))[::-1]:
        layer = network[index]
        
        loss_grad = layer.backward(inputs[index],loss_grad) #grad w.r.t. input, also weight updates
        
    return np.mean(loss)

def epochs(X, Y, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(X))
    for i in range(0, len(X) - batchsize + 1, batchsize):
        if shuffle:
            result = indices[i:i + batchsize]
        else:
            result = slice(i, i + batchsize)
        yield X[result], Y[result]

for epoch in range(10):
    acc=[]
    for x_batch,y_batch in epochs(X_train,y_train,batchsize=10,shuffle=False):
        train(network,x_batch,y_batch)
    
    acc.append(np.mean(predict(network,X_test)==y_test))
    
    print("Epoch",epoch)
    print("Val accuracy:",acc[-1])
  
