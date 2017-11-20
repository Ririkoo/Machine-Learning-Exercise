# -*- coding:utf8 -*-
import jieba
import numpy as np
import csv
import os

tokenset = set() #利用set存儲token集合
chinese_punctuations = ['，', '。', '：', '；', '?','？',
    '（', '）', '「', '」', '！', '“', '”', '\n',' '] #中文標點去除

plain_text = [] #利用list存儲每一行純文本解析結果
onehot_encode = [] #one hot encoding 
x_train=[] #輸入層 訓練素材 
y_train=[] #輸出層 訓練素材 

def stopwordslist(filepath):
    """
    獲取停用詞表
	"""
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def list2file(list,file):
    """
    將list寫入文件
	"""
    fout=open(file,'w')
    for item in list:
        for i in item:
            fout.write(str(i)+' ')
        fout.write('\n')
    fout.close()

def dict2file(dict,file):
    """
    將dict寫入文件
	"""
    with open(file,'w') as f:
        w=csv.writer(f)
        w.writerow(dict.keys())
        w.writerow(dict.values())

def processfile(path):
    """
    預處理文件并將每一個文件的分詞結果輸出
	"""
    os.chdir(path)
    sourcefile=[]
    for filename in os.listdir(path): #按照文件後綴名稱標識獲取輸入文件
        if(filename[-6:]=='_1.txt'):
            sourcefile.append(filename)
    for i in range(len(sourcefile)):
        fin = open(sourcefile[i], 'r')
        # stopwords = stopwordslist('stopwords_Chinese.txt')
        fout_content=[]
        for eachLine in fin:
            tmp_line = []
            tokens = jieba.cut(eachLine) #cut方法為jieba分詞
            for word in tokens:
                if word not in chinese_punctuations:
                # if((word not in chinese_punctuations) and (word not in stopwords)):
                    tokenset.add(word)
                    tmp_line.append(word)
            fout_content.append(tmp_line)
            plain_text.append(tmp_line)
        list2file(fout_content,sourcefile[i][:2]+'2.txt')
    fin.close()



processfile('/Users/pzhe/Documents/word2vec/DATASET') #傳入訓練素材的path，預處理文件
tokenlist=list(tokenset) 
word_to_int = {w: c for c, w in enumerate(tokenset)} # 建立token word與整數標號之間的關係 dictionary存储
vectors = np.eye(np.array(tokenlist).shape[0]) # 做每個編號的one hot encoding
word_to_vec=dict(zip(tokenlist,vectors)) # 建立token word與encoding 之間的關係 dictionary存储
dict2file(word_to_int,'word_to_int_tables.txt')# 将token word與整數標號之間的關係写入文件

for eachLine in plain_text: #遍歷分詞結果，建立訓練素材的集合，以one hot encoding表示
    for i in range(0,len(eachLine)-1):
        x=word_to_vec.get(eachLine[i]) #掃描window間隔為1
        y=word_to_vec.get(eachLine[i+1])
        tmp_word2vec=[x,y]
        onehot_encode.append(tmp_word2vec)
onehot_encode,inde=np.unique(onehot_encode, axis=0 , return_index=True) #唯一化編碼，去重複
onehot_encode=onehot_encode[np.argsort(inde)]
onehot_encode=onehot_encode.astype('uint8')


for i in range(len(onehot_encode)): #提取訓練所用輸入輸出集合
    x_train.append(onehot_encode[i][0])
    y_train.append(onehot_encode[i][1])

# 相關數據寫入文件
list2file(x_train,'x.txt') 
list2file(y_train,'y.txt')
list2file(plain_text,'merge.txt')

