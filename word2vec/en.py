# -*- coding:utf8 -*-
import nltk
import os
import numpy as np

wordEngStop = nltk.corpus.stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

tokenset=set()
onehot_encode=[]
plain_text=[]

#Read content from file
fin = open('endata.txt', 'r')
for eachLine in fin:
    tmp_line=[]
    eachLine = eachLine.lower() 
    tokens = nltk.word_tokenize(eachLine)        
    for word in tokens:
        if ((not word in wordEngStop) and (not word in english_punctuations)):
            tokenset.add(word)
            tmp_line.append(word)
    plain_text.append(tmp_line)
fin.close()

#Build data structure
tokenlist=list(tokenset)
word_to_int = {w: c for c, w in enumerate(tokenset)}
vectors = np.eye(np.array(tokenlist).shape[0])
word_to_vec=dict(zip(tokenlist,vectors))
print(word_to_int)


for eachLine in plain_text:
    for i in range(0,len(eachLine)-1):
        tmp_word2vec=[word_to_vec.get(eachLine[i]),word_to_vec.get(eachLine[i+1])]
        onehot_encode.append(tmp_word2vec)

onehot_encode,inde=np.unique(onehot_encode, axis=0 , return_index=True)
print(plain_text)
print(onehot_encode[np.argsort(inde)])
