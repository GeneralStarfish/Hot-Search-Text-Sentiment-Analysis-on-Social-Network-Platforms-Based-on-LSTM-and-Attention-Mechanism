# -*- coding: utf-8 -*-


import csv
import jieba

import numpy as np
import torch
from torch import nn

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re


csvName = "T2022-08-18.csv"

def getkey(_dict,_string):
    return [k for k,v in _dict.items() if v == _string]
        

with open(csvName, 'r', newline='') as hotCsv:
    reader = csv.reader(hotCsv)
    for i,rows in enumerate(reader):
        if i == 0:
            row = rows

#print(row)
jiebaWord = []

remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

for sentence in row:
    sentence = re.findall('[\u4e00-\u9fa5]',sentence)
    sentence = ''.join(sentence)
    jiebaWord.append(str.join(" ",jieba.cut(sentence)))
    
    
with open('fulltext.txt', 'wb') as ft:
    for _row in jiebaWord:
        ft.write( _row.encode('utf-8'))
        ft.write('\n'.encode("utf-8"))


print(jiebaWord)

model = Word2Vec(
    sentences = LineSentence('fulltext.txt'),
    sg = 0,
    vector_size = 100,
    window = 10,
    min_count = 10,
    workers=8
)

print(model.wv['国家'])
print(model.wv.most_similar('国家', topn=10))

# counts = {}
#     else:

# for word in jiebaWord:
#     if len(word) == 1:
#         continue
#         counts[word] = counts.get(word, 0) + 1 
        
# items = list(counts.items())
# print(items)

# items.sort(key=lambda x: x[1], reverse=True)

# for i in range(15):
#     word, count = items[i]
#     print("{0:<5}{1:>5}".format(word, count))
    
    





































