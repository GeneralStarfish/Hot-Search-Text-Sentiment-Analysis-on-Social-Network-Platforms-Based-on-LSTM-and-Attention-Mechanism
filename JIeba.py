# -*- coding: utf-8 -*-



import csv
import jieba

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


import time

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re







####################################训练词向量######################################


with open('pos60000.txt', 'r', encoding='utf-8') as ft:
    posOrigin = ft.readlines()
    
with open('neg60000.txt', 'r', encoding='utf-8') as ft:
    negOrigin = ft.readlines()
    
jiebaWord = []

remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

#print(row)

for _sentence in posOrigin:
    _sentence = re.findall('[\u4e00-\u9fa5]',_sentence)
    _sentence = ''.join(_sentence)
    jiebaWord.append(str.join(" ",jieba.cut(_sentence)) + " \n")

with open('pos60000.txt', 'w', encoding='utf-8') as ft:    
    for _sentence in jiebaWord:
        ft.write( _sentence)



jiebaWord = []

for _sentence in negOrigin:
    _sentence = re.findall('[\u4e00-\u9fa5]',_sentence)
    _sentence = ''.join(_sentence)
    jiebaWord.append(str.join(" ",jieba.cut(_sentence)) + " \n")

with open('neg60000.txt', 'w', encoding='utf-8') as ft:    
    for _sentence in jiebaWord:
        ft.write( _sentence)
