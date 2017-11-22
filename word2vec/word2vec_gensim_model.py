# -*- coding: utf-8 -*-  

from gensim.models import word2vec
import logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('DATASET/merge.txt') 
model = word2vec.Word2Vec(sentences, size=5 , min_count=0 ,window=1)
#size：是每个词的向量维度； 
#window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
#min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
#workers：是训练的进程数

model.save('text.model')#模型存储
model.wv.save_word2vec_format('text.model.bin') #格式化存储


model['單位'] #得到单个单词的向量表示
model.most_similar(['上班']) #得到接近相似度结果
model.similarity('單位', '上班') #判断两个词汇的相似度
