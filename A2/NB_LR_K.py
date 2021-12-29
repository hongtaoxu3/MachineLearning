from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import defaultdict
import re, os
from glob import glob
import matplotlib.pyplot as plt
import random
from random import randrange

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, remove = ('headers', 'footers', 'quotes'))
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, remove = ('headers', 'footers', 'quotes'))

pipeline = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', LogisticRegression(penalty='l2'))])
pipeline.fit(twenty_train.data,twenty_train.target)
pipeline.predict(twenty_train.data)
score = pipeline.score(twenty_test.data, twenty_test.target)
print ("LR 20news Test Set Accuracy: ",score*100,"%")

def load_texts_labels_from_folders(path, folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(idx)
    return texts, np.array(labels).astype(np.int8)


names = ['neg','pos']
movie_train,movie_train_y = load_texts_labels_from_folders(f'aclImdb/train',names)
movie_test,movie_test_y = load_texts_labels_from_folders(f'aclImdb/test',names)

#for c in [0.01, 0.05, 0.25, 0.5, 1]:
#    pipeline = Pipeline([('vectorizer', CountVectorizer(binary = True)), ('tfidf', TfidfTransformer()), ('classifier', LogisticRegression(C=c))])
#    pipeline.fit(movie_train,movie_train_y)
#    pipeline.predict(movie_train)
#    score = pipeline.score(movie_test, movie_test_y)
#    print ("C = ", c,", LR IMDB Test Set Accuracy: ",score*100,"%")
pipeline = Pipeline([('vectorizer', CountVectorizer(binary = True)), ('tfidf', TfidfTransformer()), ('classifier', LogisticRegression(C=1))])
pipeline.fit(movie_train,movie_train_y)
pipeline.predict(movie_train)
score = pipeline.score(movie_test, movie_test_y)
print ("LR IMDB Test Set Accuracy: ",score*100,"%")


def process_text(text):
    s = re.sub('[^a-z\s]+',' ', text, flags=re.IGNORECASE)
    s = re.sub('(\s+)',' ',s)
    return s.lower()


class NaiveBayes:
    
    def __init__(self,unique_classes):
        self.classes=unique_classes


    def token(self,example,i):
        if isinstance(example,np.ndarray): example=example[0]
        for word in example.split():
            self.dicts[i][word]+=1
            
    def fit(self,data,labels):
        self.examples=data
        self.labels=labels
        self.dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
        for index,c in enumerate(self.classes):
            all = self.examples[self.labels==c]
            processed=[process_text(e) for e in all]
            processed=pd.DataFrame(data=processed)
            np.apply_along_axis(self.token,1,processed,index)
            
        probs=np.empty(self.classes.shape[0])
        words=[]
        counts=np.empty(self.classes.shape[0])
        for index,c in enumerate(self.classes):
            probs[index]=np.sum(self.labels==c)/float(self.labels.shape[0])
            count=list(self.dicts[index].values())
            counts[index]=np.sum(np.array(list(self.dicts[index].values())))+1
            words+=self.dicts[index].keys()
        
        self.vocab=np.unique(np.array(words))
        self.vocab_len=self.vocab.shape[0]
        arr=np.array([counts[index]+self.vocab_len+1 for index,c in enumerate(self.classes)])
        self.info=[(self.dicts[index],probs[index],arr[index]) for index,c in enumerate(self.classes)]
        self.info=np.array(self.info)
                                              
                                              
    def getProb(self,test):
        likelihood_prob=np.zeros(self.classes.shape[0])
        for index,c in enumerate(self.classes):    
            for token in test.split():
                counts=self.info[index][0].get(token,0)+1
            
                token_prob=counts/float(self.info[index][2])
                likelihood_prob[index]+=np.log(token_prob)
                                              
        prob_arr=np.empty(self.classes.shape[0])
        for index,c in enumerate(self.classes):
            prob_arr[index]=likelihood_prob[index]+np.log(self.info[index][1])
      
        return prob_arr
    
   
    def predict(self,test):
        preds=[]
        for t in test:
            processed=process_text(t)
            prob=self.getProb(processed)
            preds.append(self.classes[np.argmax(prob)])
        return np.array(preds)

    def eva_acc(self, predict_y, real_y):
        acc=0
        for i in range(len(predict_y)):
            if(predict_y[i]==real_y[i]):
                acc=acc+1
        return acc/len(predict_y)*100.0

nb=NaiveBayes(np.unique(twenty_train.target))
nb.fit(twenty_train.data,twenty_train.target)
test_acc=nb.eva_acc(nb.predict(twenty_test.data),twenty_test.target)
print ("NB 20news Test Set Accuracy: ",test_acc,"%")

nb=NaiveBayes(np.unique(movie_train_y))
nb.fit(movie_train,movie_train_y)
test_acc=nb.eva_acc(nb.predict(movie_test),movie_test_y)
print ("NB IMDB Test Set Accuracy: ",test_acc,"%")

#def visualize_dataset(ds):
#
#    plot_X = np.arange(20, dtype=np.int16)
#    plot_Y = np.zeros(20)
#    for i in range(len(ds.data)):
#        plot_Y[ds.target[i]] += 1
#    figure = plt.figure(figsize = (16, 10))
#    figure.suptitle('Balance of data set', fontsize=16)
#    for color in ['r', 'b', 'g', 'k', 'm']:
#        plt.bar(plot_X, plot_Y, align='center', color=color)
#    plt.xticks(plot_X, ds.target_names, rotation=25, horizontalalignment='right')
#    plt.show()
#visualize_dataset(twenty_train)

M_train=[]
for i in range(len(movie_train)):
    data1=[]
    data1.append(movie_train[i])
    data1.append(movie_train_y[i])
    #print(data1)
    M_train.append(data1)
    #data_train[movie_train_y[i]]=movie_train[i]

M_test=[]
for i in range(len(movie_test)):
    data1=[]
    data1.append(movie_train[i])
    data1.append(movie_train_y[i])
    #print(data1)
    M_test.append(data1)

#data for 20news
news_train=[]
for i in range(len(twenty_train.data)):
    data1=[]
    data1.append(twenty_train.data[i])
    data1.append(twenty_train.target[i])
    #print(data1)
    news_train.append(data1)
    #data_train[movie_train_y[i]]=movie_train[i]

news_test=[]
for i in range(len(twenty_test.data)):
    data1=[]
    data1.append(twenty_test.data[i])
    data1.append(twenty_test.target[i])
    #print(data1)
    news_test.append(data1)


from random import randrange
def cross_validation_split(dataset, k, size):
        data=random.sample(dataset,size)
        result_data=list()
        ds=list(data)
        size = int(len(data)/k)
        for _ in range(k):
            eachfold=list()
            while len(eachfold) < size:
                index=randrange(len(ds))
                eachfold.append(ds.pop(index))
            result_data.append(eachfold)
        return result_data


def KfoldCV(data, K, size, algorithm):
    ds=cross_validation_split(data,K, size)
    ds1=ds
    movie_acc=[]
    num=randrange(5) 
    test=ds1.pop(num)
    train=ds
    #print(train[1][1])
    Y_test_set =[]
    X_test_set =[]
    for each in test:
        Y_test_set.append(each[1])
        X_test_set.append(each[0])
    #print(Y_test_set)
    if(algorithm =='NaiveBayes') : 
        test_acc=[]
        for i in range(len(train)):
            Y_train_set = []
            X_train_set=[]
            
            for j in range(len(train[i])):
                Y_train_set.append(train[i][j][1])
                X_train_set.append(train[i][j][0])
            #print(X_train_set)
        
            nb=NaiveBayes(np.unique(Y_train_set))
            nb.fit(X_train_set,Y_train_set)
            pclasses=nb.predict(X_test_set)
            score=nb.eva_acc(pclasses, Y_test_set)
            test_acc.append(score)
        #print(sum(test_acc)/len(train))
    #test_acc=sum(pclasses==Y_test_set)/float(len(Y_test_set))                  
        return (sum(test_acc)/len(train))
    elif(algorithm =='LR'):
        total=[]
        for i in range(len(train)):
            Y_train_set = []
            X_train_set=[]
            for j in range(len(train[i])):
                Y_train_set.append(train[i][j][1])
                X_train_set.append(train[i][j][0])
            pipeline = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', LogisticRegression(penalty='l2', C=0.5))])
            pipeline.fit(X_train_set,Y_train_set)
            pipeline.predict(X_train_set)
            score = pipeline.score(X_test_set, Y_test_set)
            total.append(score)
        
        #print(sum(total)/len(train))
        return (sum(total)/len(train))
        
#K-CV for different size of two dataasets by using different models
acc1=KfoldCV(news_train, 5, 2262, 'NaiveBayes')
print('Naive Bayes 20% 20news data size accuracy: ', acc1, '%')
acc2=KfoldCV(news_train, 5, 4523, 'NaiveBayes')
print('Naive Bayes 40% 20news data size accuracy: ', acc2, '%')
acc3=KfoldCV(news_train, 5, 6789, 'NaiveBayes')
print('Naive Bayes 60% 20news data size accuracy: ', acc3, '%')
acc4=KfoldCV(news_train, 5, 9052, 'NaiveBayes')
print('Naive Bayes 80% 20news data size accuracy: ', acc4, '%')
#acc5=KfoldCV(news_train, 5, 11314, 'NaiveBayes')
#print('Naive Bayes 100% 20news data size accuracy: ', acc5, '%')


acc11=KfoldCV(news_train, 5, 2262, 'LR')
print('Logistic Regression 20% 20news data size accuracy: ', acc11*100.0, '%')
acc22=KfoldCV(news_train, 5, 4523, 'LR')
print('Logistic Regression 40% 20news data size accuracy: ', acc22*100.0, '%')
acc33=KfoldCV(news_train, 5, 6789, 'LR')
print('Logistic Regression 60% 20news data size accuracy: ', acc33*100.0, '%')
acc44=KfoldCV(news_train, 5, 9052, 'LR')
print('Logistic Regression 80% 20news data size accuracy: ', acc44*100.0, '%')
#acc55=KfoldCV(news_train, 5, 11314, 'LR')
#print('Logistic Regression 100% 20news data size accuracy: ', acc55*100.0, '%')



import matplotlib.pyplot as plt
#x_axis=['20% data','40% data', '60% data', '80% data', '100% data']
#y_axis=[acc1, acc2, acc3, acc4, acc5]
x_axis=['20% data','40% data', '60% data', '80% data']
y_axis=[acc1, acc2, acc3, acc4]
plt.plot(x_axis,y_axis, '-b', label='Naive Bayes')

#x_axis=['20% data','40% data', '60% data', '80% data', '100% data']
#yaxis=[acc11*100.0, acc22*100.0, acc33*100.0, acc44*100.0, acc55*100.0]
x_axis=['20% data','40% data', '60% data', '80% data']
yaxis=[acc11*100.0, acc22*100.0, acc33*100.0, acc44*100.0]
plt.plot(x_axis, yaxis, '-r', label='Logistic Regression')
plt.title('5-fold-CV of 20news dataset')
plt.xlabel('different size of data')
plt.ylabel('accuracy')
plt.legend()
plt.show()


ac1=KfoldCV(M_train, 5, 5000, 'LR')
print('Logistic Regression 20% movie data size accuracy: ', ac1*100.0, '%')
ac2=KfoldCV(M_train, 5, 10000, 'LR')
print('Logistic Regression 40% movie data size accuracy: ', ac2*100.0, '%')
ac3=KfoldCV(M_train, 5, 15000, 'LR')
print('Logistic Regression 60% movie data size accuracy: ', ac3*100.0, '%')
ac4=KfoldCV(M_train, 5, 20000, 'LR')
print('Logistic Regression 80% movie data size accuracy: ', ac4*100.0, '%')
#ac5=KfoldCV(M_train, 5, 25000, 'LR')
#print('Logistic Regression 100% movie data size accuracy: ', ac5*100.0, '%')

ac11=KfoldCV(M_train, 5, 5000, 'NaiveBayes')
print('Naive Bayes 20% movie data size accuracy: ', ac11, '%')
ac22=KfoldCV(M_train, 5, 10000, 'NaiveBayes')
print('Naive Bayes 40% movie data size accuracy: ', ac22, '%')
ac33=KfoldCV(M_train, 5, 15000, 'NaiveBayes')
print('Naive Bayes 60% movie data size accuracy: ', ac33, '%')
ac44=KfoldCV(M_train, 5, 20000, 'NaiveBayes')
print('Naive Bayes 80% movie data size accuracy: ', ac44, '%')
#ac55=KfoldCV(M_train, 5, 25000, 'NaiveBayes')
#print('Naive Bayes 100% movie data size accuracy: ', ac55, '%')

import matplotlib.pyplot as plt
#x_axis1=['20% data','40% data', '60% data', '80% data', '100% data']
#y_axis1=[ac1*100.0, ac2*100.0, ac3*100.0, ac4*100.0, ac5*100.0]
x_axis1=['20% data','40% data', '60% data', '80% data']
y_axis1=[ac1*100.0, ac2*100.0, ac3*100.0, ac4*100.0]
plt.plot(x_axis1,y_axis1,"-r", label='Logistic Regression' )

#yaxis1=[ac11, ac22, ac33, ac44, ac55]
#x_axis1=['20% data','40% data', '60% data', '80% data', '100% data']
yaxis1=[ac11, ac22, ac33, ac44]
x_axis1=['20% data','40% data', '60% data', '80% data']
plt.plot(x_axis1, yaxis1, '-b', label='Naive Bayes')
plt.title('5-fold-CV of IMDB movie dataset')
plt.xlabel('different size of data')
plt.ylabel('accuracy (%)')
plt.legend()
plt.show()




