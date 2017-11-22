# -*- coding:utf8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_train=[] #input layer訓練資料集
y_train=[] #output layer訓練資料集
word_to_int={} #token word與編號的關係

def sigma(vector, deriv=False):
    if deriv:
        return sigma(vector)*(1-sigma(vector))
    else:
        return np.exp(vector)/np.exp(vector).sum()

def getdata(path):
    """
    從文本讀入所需數據
	"""
    global x_train,y_train,word_to_int
    os.chdir(path)
    with open('x.txt','r') as f:
        for line in f:
            tmpline=line.replace("\n",'')
            x_train.append(np.array(list(tmpline.replace(" ",""))).astype('int'))
        f.close()
    with open('y.txt','r') as f:
        for line in f:
            tmpline=line.replace("\n",'')
            y_train.append(np.array(list(tmpline.replace(" ",""))).astype('int'))
        f.close()
    with open('word_to_int_tables.txt','r') as f:
        cnt=0
        for line in f:
            if(cnt==0):
                tmpline=line.replace("\n","")
                tmpkey=tmpline.split(',')
            else:
                tmpline=line.replace("\n","")
                tmpval=tmpline.split(',')
                tmpval = [ int(x) for x in tmpval ]
            cnt+=1
        word_to_int=dict(zip(tmpkey,tmpval))
    f.close()


getdata('/Users/pzhe/Documents/word2vec/DATASET')

#一些訓練參數的設定
log_likelihood = np.array([])
epochs = 10000 #iteration次數
learning_rate = 0.005 #learning_rate
tolerance = 0.001 #若loss變化小於tolerance 終止訓練
discount = float(learning_rate)/epochs
H = 2 #hidden layer的dimension

#訓練開始
print('Start training...')
U = np.random.randn(H,len(x_train[0])) #隨機分配權重矩陣
V = np.random.randn(len(x_train[0]),H) 
for epoch in range(epochs): 
    likelihood = 0
    for i in range(len(x_train)): #每一次batch為整個x dataset
        # Forward propogate word 
        # Model Updating Rule: gradient descent
        l_input = x_train[i] #Input layer
        l_hidden = np.dot(U,l_input) #hidden layer
        l_output = np.dot(V,l_hidden) #output layer
        l_output_a = sigma(l_output)  #output layer(normalization)
        errors = np.zeros(len(x_train[0]))
        # 計算誤差，修正權重矩陣
        l_target= y_train[i]
        errors += (l_output_a-l_target)
        delta2 = errors*sigma(l_output,True)
        V -= learning_rate*np.outer(delta2,l_hidden)
        U -= learning_rate*np.outer(np.dot(V.T,delta2),l_input)
        likelihood+=sum(map(np.log,l_output_a))
    log_likelihood=np.append(log_likelihood,likelihood)
    learning_rate -= discount 
    if epoch<2: continue
    if (abs(likelihood-log_likelihood[-2])<tolerance):
        break
print('Stop training.')

print (len(log_likelihood))
plt.scatter(U[0],U[1], alpha=0.3)
for key, value in word_to_int.items():
    plt.text(U[0][value],U[1][value],key, size=8)
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.show()#畫出分布（只限二維情況）
plt.title('Plot of likelihood vs. iteration times')
plt.xlabel('iteration times')
plt.ylabel('likelihood')
plt.plot(log_likelihood) #畫出loss(log likehood)
plt.show()
