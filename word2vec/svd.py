# -*- coding:utf8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

text=[]
#參照http://cpmarkchang.logdown.com/posts/772665-nlp-vector-space-semantics實現

def getdata(path):
    """
    從文本讀入所需數據
	"""
    os.chdir(path)
    fin = open('merge.txt', 'r')
    # stopwords = stopwordslist('stopwords_Chinese.txt')
    for eachLine in fin:
        tmp_line = []
        tmpline=eachLine.replace("\n",'')
        tokens = tmpline.split(" ")
        for word in tokens:
            if(word!=""):
                tmp_line.append(word)
        text.append(tmp_line)
    fin.close()

def build_word_vector(text):
    """
    建立word2vector資料結構
	"""
    word2id = {w: i for i, w in enumerate(sorted(list(set(reduce(lambda a, b: a + b, text)))))}
    id2word = {x[1]: x[0] for x in word2id.items()}
    wvectors = np.zeros((len(word2id), len(word2id)))
    for sentence in text:
        for word1, word2 in zip(sentence[:-1], sentence[1:]):
            id1, id2 = word2id[word1], word2id[word2]
            wvectors[id1, id2] += 1
            wvectors[id2, id1] += 1
    return wvectors, word2id, id2word


def cosine_sim(v1, v2):
    """
    計算兩個向量的距離
	"""
    return np.dot(v1, v2) / (np.sqrt(np.sum(np.power(v1, 2))) * np.sqrt(np.sum(np.power(v1, 2))))


def visualize(wvectors, id2word):
    """
    可視化結果（SVD二維）
	"""
    np.random.seed(10)
    fig = plt.figure()
    U, sigma, Vh = np.linalg.svd(wvectors)

    ax = fig.add_subplot(111)
    ax.axis([-1, 1, -1, 1])
    for i in id2word:
        ax.scatter(U[i,0],U[i,1], alpha=0.3)
        ax.text(U[i, 0], U[i, 1], id2word[i], alpha=0.8, size=9)
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False 
    plt.show()

getdata('/Users/pzhe/Documents/word2vec/DATASET')
wvectors, word2id, id2word = build_word_vector(text)
visualize(wvectors, id2word)