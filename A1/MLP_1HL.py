import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from random import seed
from random import random
import tensorflow as tf
import time
import copy

class MLP1():
    layer1={}
    layer2={}

    def __init__(self, inputsize, hiddensize, outputsize):
      self.input=inputsize
      self.hidden=hiddensize
      self.output=outputsize

      self.layer1['W'] =np.random.randn(hiddensize,inputsize) / np.sqrt(hiddensize)
      self.layer1['b'] =np.random.randn(hiddensize, 1)/np.sqrt(hiddensize)
      self.layer2['W'] = np.random.randn(outputsize, hiddensize)/np.sqrt(outputsize)
      self.layer2['b']=np.random.randn(outputsize,1)/np.sqrt(outputsize)

    def function(self, x,type, deri=False):
      if type == 'ReLU':
            if deri == True:
                return np.array([1 if i>0 else 0 for i in np.squeeze(x)])
            else:
                return np.array([i if i>0 else 0 for i in np.squeeze(x)])
      elif type == 'Sigmoid':
            if deri == True:
                return 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))
            else:
                return 1/(1+np.exp(-x))
    
       
    def softmax(self,x):
        return 1/sum(np.exp(x)) * np.exp(x)
    
    def entropy_error(self,v,y):
        return -np.log(v[y])

    def forward(self, x, y):
        Z = np.matmul(self.layer1['W'],x).reshape((self.hidden,1)) + self.layer1['b']
        H = np.array(self.function(Z, 'ReLU')).reshape((self.hidden,1))
        U = np.matmul(self.layer2['W'],H).reshape((self.output,1)) + self.layer2['b']
        predict_list = np.squeeze(self.softmax(U))
        error = self.entropy_error(predict_list,y)
        dict={'Z':Z, 'H':H, 'U':U, 'function':predict_list.reshape((1, self.output)), 'error':error }
        return dict

    def backward(self, x, y, result):
     grad = np.array([0]*self.output).reshape((1,self.output))
     grad[0][y]=1
     deri_U = (-(grad - result['function'])).reshape((self.output,1))
     cp = copy.copy(deri_U)
     dc = np.matmul(deri_U, result['H'].transpose())
     delta = np.matmul(self.layer2['W'].transpose(), deri_U)
     db1 = delta.reshape(self.hidden,1)*self.function(result['Z'],'ReLU',deri=True).reshape(self.hidden,1)
     dW=np.matmul(db1.reshape(self.hidden, 1), x.reshape((1, 784)))

     graddict={'dc':dc, 'db2':cp, 'db1':db1, 'dw':dW}

     return graddict

    def update(self, result, l_rate):
     self.layer1['W'] -= result['dw']*l_rate
     self.layer1['b'] -=result['db1']*l_rate
     self.layer2['W'] -= result['dc']*l_rate
     self.layer2['b'] -= result['db2']*l_rate

    def loss (self, x, y):
     loss=0
     for i in range(len(x)):
       x1=x[i][:]
       y1=y[i]
       loss += self.forward(x1, y1)['error']
     return loss

    def eva_acc(self,X_test, Y_test):
        total_correct = 0
        for n in range(len(X_test)):
            y = Y_test[n]
            x = X_test[n][:]
            prediction = np.argmax(self.forward(x,y)['function'])
            if (prediction == y):
                total_correct += 1
        return total_correct/np.float(len(X_test))
 
    def predict(self, X_train, Y_train, x_test, y_test,  epochs=10, learning_rate = 0.01):
        rand_indices = np.random.choice(len(X_train), 10000, replace=True)
        
        ep=1
        count = 1
        loss_dict = {}
        test_dict = {}
        
        for i in rand_indices:

            f_result = self.forward(X_train[i],Y_train[i])
            b_result = self.backward(X_train[i],Y_train[i],f_result)
            self.update(b_result,0.01)
            
            if count % 1000 == 0:
                   loss = self.loss(X_train,Y_train)
                   test = self.eva_acc(x_test,y_test)
                   print('Trained for {} times,'.format(ep),'accuracy = {}'.format(test*100.0))
                   loss_dict[str(count)]=loss
                   test_dict[str(count)]=test
                   ep+=1
            count += 1
        print('Training finished!')

(x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()
x_train_norm = x_train1.reshape(60000, 784)
x_test_norm = x_test1.reshape(10000, 784)
def normalize(dataset):
  return dataset/255.0
x_train_norm = x_train_norm.astype('float32') / 255.
x_test_norm = x_test_norm.astype('float32') / 255.
y_test_n = y_test1
y_train_n = y_train1
num_iterations = 100
# set the base learning rate
learning_rate = 0.01
# number of inputs
num_inputs = 28*28
# number of outputs
num_outputs = 10
# size of hidden layer
hidden_size = 128
model = MLP1(784,128,10)
accu = model.predict(x_train_norm,y_train_n,x_test_norm, y_test_n, 10 ,0.01)
