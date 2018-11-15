import numpy as np
import pickle
import torch
import random
import torch.nn.functional as F
from numpy import linalg as la
from torch.autograd import Variable
import scipy.sparse as sparse
from torch import nn
from gensim.models import word2vec
from collections import Counter
import gensim
#读入数据
with open('data/binarygraph.pickle', 'rb') as file:#二分图，从起点到终点的有向边，边要大于60
    binarygraph = pickle.load(file)
with open('data/myvec.pickle', 'rb') as file:#每个点对应的向量，一共128维，前64维为起点向量，后64维为终点向量
    vecmat = pickle.load(file)


# # 常用函数
def isnan(num):#判断是不是nan
    return num != num
flag = np.zeros(24)
flag[6:12] = 1
flag[12:18] = 2
flag[18:] = 3
def trans(ttstr):#转换时间
    m = int(ttstr.split(' ')[1].split(':')[0])
    return str(int(flag[m]))
def get_accnum(output, label):#分类的准确率函数
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return (num_correct , total)

#每个点的起点终点向量单独表示
vocab = []
for m in vecmat.keys():
    vocab.append('a'+m)
    vocab.append('b'+m)
vocab2dig = dict(zip(vocab,range(len(vocab))))#词汇2数字转换表
dig2vocab = dict(zip(range(len(vocab)),vocab))#转换表2词汇
vocabvec = np.zeros((len(vocab),64))
for key in vocab:
    if key[0] == 'a':
        vocabvec[vocab2dig[key]] = vecmat[key[1:]][:64]
    else:
        vocabvec[vocab2dig[key]] = vecmat[key[1:]][64:]
vocabvec = torch.tensor(vocabvec).float().cuda()
time2dig = {'a0':0,'a1':1,'a2':2,'a3':3}
#初始化时间向量，作为参数参与网络训练
timevec = torch.rand(4,64).cuda()
timevec.requires_grad = True


# # 通过时间属性预测,做训练的数据集

#正样本跟负样本的数据集
posdata = {}#正样本数据 (时间:[起点集合，终点集合])
negdata = {}#负样本数据(时间：终点集合（用于随机取数）)
relabel = {}
for eg in binarygraph.keys():
    eglist = eg.split(' ')
    if 'a'+eglist[0] not in vocab or 'b'+eglist[1] not in vocab:
        continue
    s = vocab2dig['a'+eglist[0]]
    e = vocab2dig['b'+eglist[1]]
    t = time2dig[eglist[2]]
    #正样本数据
    if t not in posdata.keys():
        posdata[t] = [[],[]]#表示的是起点list跟终点list
        relabel[t] = []
    for i in range(int(binarygraph[eg]/60)):
        posdata[t][0].append(s)
        posdata[t][1].append(e)
        m = min(10,int(binarygraph[eg]/120))
        relabel[t].append(m)
    #负样本数据
    if t not in negdata.keys():
        negdata[t] = []#终点list
    for i in range(int(binarygraph[eg]/60)):
        negdata[t].append(e)
for key in posdata.keys():
    posdata[key][0] = np.array(posdata[key][0])
    posdata[key][1] = np.array(posdata[key][1])
    relabel[key] = torch.tensor(np.array(relabel[key])).long().cuda()
    negdata[key] = np.array(negdata[key])


# # 预测模型
class predictnet(nn.Module):
    def __init__(self):
        super(predictnet, self).__init__()
        self.block1 = nn.Linear(192,400)
        self.block2 = nn.Linear(400,400)
        self.classifier1 = nn.Linear(400, 2)
        self.classifier2 = nn.Linear(400, 11)
    def forward(self, x,y,t):
        input1 = torch.cat((x,y,t),1)
        hidden2 = F.relu(self.block1(input1))
        hidden3 = F.relu(self.block2(hidden2))
        out1 = self.classifier1(hidden3)#二分类
        out2 = self.classifier2(hidden3)#十一分类
        return (out1,out2)
    
net = predictnet().cuda()
loss_fn = torch.nn.CrossEntropyLoss()#多分类的损失函数
optimizer = torch.optim.Adam(list(net.parameters())+[timevec], lr=3e-3,betas=(0.5, 0.999))


# # 训练模型
import copy
epoch = 800
num = 10000
#生成索引id，便于shuffle跟索引原始的值
posid= {}
negid = {}
for key in posdata.keys():
    posid[key] = list(range(posdata[key][0].shape[0]))#正样本数据集，起点跟终点要公用一个id
    negid[key] = list(range(negdata[key].shape[0]))
#开始训练
for i in range(epoch):
    #先进行shuffle
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for key in posid.keys():
        random.shuffle(posid[key])
        random.shuffle(negid[key])
    #提取数据,这里每轮丢进去1k条正样本与1k条负样本
    for time in posid.keys():#每个时间段分开训练
        startid = 0#起始的数据id点
        while(startid < len(posid[time])):
            endid = min(len(posid[time]) , num + startid)
            psdata = posdata[time][0][posid[time][startid:endid]]#这次训练的正样本起点
            pedata = posdata[time][1][posid[time][startid:endid]]#这次训练的正样本终点
            nedata = negdata[time][negid[time][startid:endid]]#这次训练的负样本终点
            psve = vocabvec[psdata]#正起点矩阵
            peve = vocabvec[pedata]#正终点矩阵
            neve = vocabvec[nedata]#负终点矩阵
            timeve = timevec[time].expand(endid-startid,64)#时间矩阵
            (pout1,pout2) = net(psve,peve,timeve)#正output
            (nout1,nout2) = net(psve,neve,timeve)#负output
            plabel1 = torch.ones(endid-startid).long().cuda()#真实标签,第一个分类器
            plabel2 = relabel[time][posid[time][startid:endid]]#真实的标签，第二个分类器
            nlabel = torch.zeros(plabel1.shape).long().cuda() * 1#伪标签
            data = torch.cat((pout1,nout1),0)
            label = torch.cat((plabel1,nlabel),0)
            (num_correct , total) = get_accnum(data, label)
            s1 += num_correct
            s2 += total
            (num_correct , total) = get_accnum(pout2,plabel2)
            s3 += num_correct
            s4 += total
            loss1 = loss_fn(data,label)  #第一个分类器的损失
            loss2 = loss_fn(pout2,plabel2) #第二个分类器的损失
            loss = 4*loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 优化判别网络
            startid = endid
    if (i+1)%100 == 0:
        print((i+1),loss1.item(),loss2.item(),loss.item(),s1,s2,s1/s2,s3,s4,s3/s4)
    if (i+1)%100 == 0 and s3/s4 >= 0.98 and s1/s2>=0.93:
        Net = copy.deepcopy(net)
        tv = copy.deepcopy(timevec)
        torch.save(Net.cpu(), 'data/muti_model_'+str(i+1)+'.pkl')
        np.save('data/timevec_'+str(i+1) + '.npy', tv.cpu().detach().numpy())