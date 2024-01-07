# -*- coding: utf-8 -*-

import csv
import jieba

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import math

from gensim.models import Word2Vec

from gensim.models.word2vec import LineSentence
import re

import dill.source


sentenceLen = 20 #句子长度

vector_size_ = 100 #词向量维度

train_num = 40000

batch_size = 200




iftrain = 2

WVmin_count = 5

WVwindow = 10

WVworkers = 8

_epoch = 10


#####################################分词导入######################################

csvName = "2022-08-18.csv"

inputRow = 15

def getkey(_dict,_string):
    return [k for k,v in _dict.items() if v == _string]
        

with open(csvName, 'r', newline='') as _Csv:
    reader = csv.reader(_Csv)
    for i,_row in enumerate(reader):
        if i == inputRow:
            row = _row
            
jiebaWord = []

remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

#print(row)

for _sentence in row:
    _sentence = re.findall('[\u4e00-\u9fa5]',_sentence)
    _sentence = ''.join(_sentence)
    jiebaWord.append(str.join(" ",jieba.cut(_sentence)) + " \n")
    

#####################################函数部分#######################################


def trainWord2Vec(_origin):

    with open('trainCutStopword.txt', 'w', encoding='utf-8') as ft:    
        for _row in _origin:
            ft.write( _row)

    _wModel = Word2Vec(
        sentences = LineSentence('trainCutStopword.txt'),
        sg = 0,
        vector_size = vector_size_,
        window = WVwindow,
        min_count = WVmin_count,
        workers=WVworkers
        )

    return _wModel
    
def updateWord2Vec(_origin,_new,_originModel):
    
    with open('trainCutStopwordUnlabel.txt', 'w', encoding='utf-8') as ft:    
        for _row in _new:
            ft.write( _row)
            
        
    _wModel = Word2Vec(
        sg = 0,
        vector_size = vector_size_,
        window = WVwindow,
        min_count = WVmin_count,
        workers=WVworkers
    )
    
    _Lnew = LineSentence('trainCutStopwordUnlabel.txt')
    
    _wModel.build_vocab(_Lnew)
    _wModel.wv.vectors_lockf = np.ones(len(_wModel.wv))
    _wModel.wv.intersect_word2vec_format(fname=_originModel)
    _wModel.train(_Lnew,epochs=5,total_examples=_wModel.corpus_count)
    
    
    print('finish update')
    
    return _wModel
    
###############################训练模型###############################


def senTo200(_sentence):#全部转化为固定长度
    _rowList = []
    for _word in _sentence.split():
        #print(_word) 
        if wModel.wv.has_index_for(_word) == True:
            #print(wModel.wv[_word])
            _rowList.append(list(wModel.wv[_word]))
            
    if len(_rowList) < sentenceLen:
        for i in range(0,sentenceLen-len(_rowList)) :
            _rowList.append([0]*vector_size_)
    else:
        _rowList = _rowList[0:sentenceLen]
    return _rowList




class CoreNet(nn.Module):
    
    def __init__(self):
        super(CoreNet, self).__init__()
        
        self.n_layers = n_layers = 2 # LSTM的层数
        self.hidden_dim = hidden_dim = 512 # 隐状态的维度
        drop_prob = 0.5 # dropout层概率
        self.lstm = nn.LSTM(vector_size_, # 输入的维度
                            hidden_dim, # hidden_state的维度
                            n_layers, # LSTM的层数
                            dropout=drop_prob, 
                            batch_first=True # 第一个维度是否是batch_size
                           )
        # LSTM层
        self.fc = nn.Linear(in_features=hidden_dim, # 将LSTM的输出作为线性层的输入
                            out_features=1 # 输出0或
                            ) 
        #全连接线性层
        self.sigmoid = nn.Sigmoid() 
        #sigmoid
        self.GetQ = nn.Linear(in_features=hidden_dim, out_features=hidden_dim ) 
        self.GetK = nn.Linear(in_features=hidden_dim, out_features=hidden_dim ) 
        self.GetV = nn.Linear(in_features=hidden_dim, out_features=hidden_dim )  
        #注意力机制 Q K V       
        self.dropout = nn.Dropout(drop_prob)
        #Dropout

    def attention(self,Q,K,V):
        d_k= K.size(-1)
        scores = torch.matmul(Q,K.transpose(1,2))/math.sqrt(d_k)
        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)
        output = context.sum(1)
        
        return output,alpha_n
    #attention层

    def forward(self, x, hidden):
        
        lstm_out, hidden = self.lstm(x, hidden)    
        
        out = self.dropout(lstm_out)
        
        Q = self.GetQ(out)
        K = self.GetK(out)
        V = self.GetV(out)
        
        attn_output,alpha_n = self.attention(Q,K,V)
        
        out = self.fc(attn_output)

        out = self.sigmoid(out)

        # 取最后一个单词的输出
        out = out[:,-1]
   
        return out, hidden 
    
    def init_hidden(self, batch_size):

        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
                 )
        return hidden




def train(_model,_train_loader,epochs = _epoch ,counter = 0,print_every = 10):
    
    for i in range(epochs):
        h = _model.init_hidden(batch_size) # 初始化Hidden_state
        
        for inputs, labels in _train_loader: # 从train_loader中获取一组inputs和labels
            counter += 1 # 训练次数+1
            # 将上次输出的hidden_state转为tuple格式
            h = tuple([e.data for e in h]) 
            # 将数据迁移到GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清空模型梯度
            model.zero_grad()
            
            # 将本轮的输入和hidden_state送给模型，进行前向传播，
            # 然后获取本次的输出和新的hidden_state
            output, h = model(inputs, h)
            
            # 将预测值和真实值送给损失函数计算损失
            loss = criterion(output, labels.float())
            
            # 进行反向传播
            loss.backward()
            
            # 对模型进行裁剪，防止模型梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            # 更新权重
            optimizer.step()
            
            # 隔一定次数打印一下当前状态
            if counter%print_every == 0:
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()))



def test(_model,_test_loader):
    
    test_losses = [] # 记录测试数据集的损失
    num_correct = 0 # 记录正确预测的数量
    
    p_num_correct = 0
    pp_num_correct = 0
    all_p_corret = 0
    
    h = model.init_hidden(batch_size) # 初始化hidden_state和cell_state
    model.eval() # 将模型调整为评估模式（去掉dropout和一些其他的）
    
    for inputs, labels in _test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) # 将模型四舍五入为0和1
        correct_tensor = pred.eq(labels.float().view_as(pred)) # 计算预测正确的数据
        
        p_correct_tensor = pred.eq(labels.float().view_as(pred)).mul(labels.float()).view_as(pred)
        p_correct = np.squeeze(p_correct_tensor.cpu().numpy())
        p_num_correct += np.sum(p_correct)
        
        pp_correct_tensor = pred.eq(labels.float().view_as(pred)).mul(pred.float()).view_as(pred)
        pp_correct = np.squeeze(pp_correct_tensor.cpu().detach().numpy())
        pp_num_correct += np.sum(pp_correct)
        all_p_corret += np.sum(np.squeeze(pred.cpu().detach().numpy()))
        
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
        
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(_test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))
    r_test_acc = p_num_correct/len(_test_loader.dataset)*2
    
    print("r: {:.3f}%".format(r_test_acc*100))
    p_test_acc = pp_num_correct/all_p_corret
    print("p: {:.3f}%".format(p_test_acc*100))




def predict(_sentence):
    _sentence = [senTo200(_sentence )]
    
    # 将数据移到GPU中
    _sentence = torch.Tensor(_sentence).to(device)
    
    # 初始化隐状态
    h = (torch.Tensor(2, 1, 512).zero_().to(device),
         torch.Tensor(2, 1, 512).zero_().to(device))
    h = tuple([each.data for each in h])
    
    # 预测
    if model(_sentence, h)[0] >= 0.5:
        return "positive"
    else:
        return "negative"
        
    
####################################主函数########################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

######加载或训练模型########

with open('pos60000.txt', 'r', encoding='utf-8') as ft:
    posOrigin = ft.readlines()
        
with open('neg60000.txt', 'r', encoding='utf-8') as ft:
    negOrigin = ft.readlines()
        
origin = posOrigin + negOrigin         
print('finish load')
    #导入词库        

if iftrain >= 1: #是否重新训练模型
    
    if iftrain == 2:  #是否重新建立训练集
        wModel = trainWord2Vec(origin)
        wModel.save('Word2Vec.model')
        wModel.wv.save_word2vec_format('Word2VecFormat.model')
    
        print('finish Word2Vec')
        #训练Word2Vec
    
        posList = []

        for _row in posOrigin:         
            posList.append(senTo200(_row ))
        
        negList = []
        
        for _row in negOrigin:
            negList.append(senTo200(_row ))
    
        Listall = []
        Labelall = []

        for i in range(0,min(len(posOrigin),len(negOrigin))):
            Listall.append(posList[i])
            Labelall.append(1)
            Listall.append(negList[i])
            Labelall.append(0)
        
        print('finish enformat')
        #得到固定长度的训练与测试集

    
        TrainTorch = torch.Tensor(Listall[0:train_num])
        TestTorch = torch.Tensor(Listall[train_num:])
    
        TrainLabel = torch.Tensor(Labelall[0:train_num])
        TestLabel = torch.Tensor(Labelall[train_num:])    
    
        print('finish Tensor')
        #得到固定长度的训练与测试集
    
        train_data = TensorDataset(TrainTorch, TrainLabel)
        test_data = TensorDataset(TestTorch, TestLabel)
    
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
        print('finish dataloader')
        #Dataloader
        
        with open('./train_loader_save.pkl','wb') as f:
            dill.dump(train_loader, f)
        with open('./test_loader_save.pkl','wb') as f:
            dill.dump(test_loader, f)
            
        print('finish saveDataLoader')   
        #保存数据集
        
    else: #直接加载训练集
        wModel = updateWord2Vec(origin,jiebaWord,'Word2VecFormat.model')
        with open('./train_loader_save.pkl','rb') as f:
            train_loader = dill.load(f)

        with open('./test_loader_save.pkl','rb') as f:
            test_loader = dill.load(f)
        print('finish loadDataLoader')
    
    model = CoreNet()
    model.to(device)
    
    #print(model)
    
    criterion = nn.BCELoss()
    lr = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    train(model,train_loader)
    torch.save(model,"model")
    
    test(model,test_loader)    
    
else:
    
    wModel = updateWord2Vec(origin,jiebaWord,'Word2VecFormat.model')
    model = torch.load("model")

        
#####Output#####


if iftrain == 0:
    
    output = []
    posRate = 0.0
    
    for i,_sentence in enumerate(row):
            
        OutIfPos = predict(jiebaWord[i])
        if OutIfPos == "positive" :
            posRate +=1
            
        _showRealInPut = []
        
        for _word in jiebaWord[i].split():
            if wModel.wv.has_index_for(_word) == True:
                _showRealInPut.append(_word)
        
        if len(_showRealInPut) > sentenceLen:
            _showRealInPut = _showRealInPut[0:sentenceLen]
    
        _showRealInPut = str.join(" ",_showRealInPut)+"\n"
    
    
        output.append(_showRealInPut + "      " + OutIfPos + "\n")
        
    posRate = posRate/len(row)
    
    with open('Output.txt', 'w', encoding='utf-8') as ft:    
        ft.write("标题： " + row[0] + "\n赞同率：" + str(posRate) + "\n\n")
        for _row in output:
            ft.write( _row)


