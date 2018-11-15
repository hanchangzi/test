
# coding: utf-8

# In[8]:


# coding: utf-8
# # 测试集中的数据出现在训练集中的***
import numpy as np
from torch import nn
import random
import bisect
import torch.nn.functional as F
import pickle
import torch
import math
from numpy import linalg as la
from torch.autograd import Variable
import scipy.sparse as sparse
from torch import nn
import gensim
from collections import Counter
from gensim.models import word2vec
##读入数据
with open('data/test_data.pickle', 'rb') as file:#测试数据
    test_data = pickle.load(file)
with open('data/train_data.pickle', 'rb') as file:#训练数据
    train_data = pickle.load(file)
timevec = torch.tensor(np.load('data1/timevec_300.npy')).float()#时间向量
tvec = timevec.detach().numpy()#将gpu tensor提出来为numpy
with open('data/orgraph.pickle', 'rb') as file:#原始的无向出行网络图，边大于等于60单
    orgraph = pickle.load(file)
with open('data/myvec.pickle', 'rb') as file:#区域embedding的向量
    vecmat = pickle.load(file)
vocab = vecmat.keys()#区域id的集合
##载入网络模型
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
net = predictnet()
net = torch.load('data1/muti_model_300.pkl')
##常用函数以及数据定义
#March跟April中休息日被标记为1，工作日被标记为0
March = np.zeros(32)
March[:5] = 1
March[5:12] = 2
March[12:19] = 3
March[19:26] = 4
March[26:] = 5
April = np.zeros(31)
April[:2] = 5
April[2:9] = 6
April[9:16] = 7
April[16:23] = 8
#将一天划分为4个时间段
flag = np.zeros(24)
flag[6:12] = 1
flag[12:18] = 2
flag[18:] = 3
def weight_delay(stime):#权重衰减函数
    date =  stime.split(' ')[0].split('-')
    month = int(date[1])
    day = int(date[2])
    if month == 3:
        week = March[day]
    else:
        week = April[day]
    return np.sqrt(1-(8-week)*(8-week)/64)
def isnan(num):#判断是不是NAN
    return num != num
def gettime(ttstr):#转换时间，将24小时映射到4个时间段内
    m = int(ttstr.split(' ')[1].split(':')[0])
    return int(flag[m])


usrlist = list(train_data.keys())#用户列表
usr2dig = dict(zip(usrlist,range(len(usrlist))))


udd = {}#用户走过的休闲娱乐的点
uweek = {}#用户走过的休闲娱乐的点权值
ucom = {}#休闲娱乐场场景下，用户的非休闲娱乐点
uedge = {}#用户走过的边的集合
otherpoint = ['办公地点','住宅地名','教育']
usr2list = []#休闲娱乐的次数少于2单的用户
usr1list= []#休闲娱乐的次数少于1单的用户
ucomt = {}#每个人自己选中的另一个点的出行时间规律,key是uid
ucomtvec = {}#每个人另一个点的时间的向量
def getucom(um,umwe):
    df = {}
    for i in range(len(um)):
            if  um[i] not in df:
                df[um[i]] = 0
            df[um[i]] += umwe[i]
    result = sorted(df.items(), key=lambda item:item[1], reverse=True)
    return [m[0] for m in result][:2]

for usr in usrlist:
    recorddata = train_data[usr]
    uid = usr2dig[usr]
    uweek[uid] = []
    udd[uid] = []
    ucom[uid] = []
    uedge[uid] = []
    um = []
    umwe = []
    for i in range(len(recorddata)):
       record = recorddata[i]
       uedge[uid].append(record[0])
       pp = record[0].split(' ')
       #获取uid的休闲娱乐点，并且记录其权重
       if isnan(record[4]) == False and record[4].split(':')[0]=='休闲娱乐' and record[6] =='休闲娱乐':
            udd[uid].append(pp[0])
            uweek[uid].append(weight_delay(record[1]))
       if isnan(record[5]) == False and record[5].split(':')[0]=='休闲娱乐' and record[6] =='休闲娱乐':
            udd[uid].append(pp[1])
            uweek[uid].append(weight_delay(record[1]))
        #获取该用户走过的教育/家庭/公司的点，并记录其权重
       if isnan(record[4]) == False and record[4].split(':')[0] in otherpoint:
            um.append(pp[0] + ' ' + record[4].split(':')[0])
            umwe.append(weight_delay(record[1]))
            continue
       if isnan(record[5])==False and record[5].split(':')[0] in otherpoint:
            um.append(pp[1] + ' ' + record[5].split(':')[0])
            umwe.append(weight_delay(record[1]))
            continue
    ucom[uid] = getucom(um,umwe)#确定另一个点
    udd[uid].reverse()
    uweek[uid].reverse()
    uedge[uid] = list(set(uedge[uid]))
    d = len(udd[uid])
    if d < 2:
        usr2list.append(usr)
    if d<1:
        usr1list.append(usr)
    #------------------用户另一个点出行时间的统计----------------
    com = ucom[uid]
    ucomt[uid] = {}
    for cm in com:
       ucomt[uid][cm] = np.zeros(4)
       for i in range(len(recorddata)):
           record = recorddata[i]
           pp = record[0].split(' ')
           if cm.split(' ')[0]==pp[0] and isnan(record[4])==False and record[4].split(':')[0]== cm.split(' ')[1]:
                ucomt[uid][cm][gettime(record[1])]+=weight_delay(record[1])
           if cm.split(' ')[0]==pp[1] and isnan(record[5])==False and record[5].split(':')[0]== cm.split(' ')[1]:
                ucomt[uid][cm][gettime(record[1])] += weight_delay(record[1])
                


# In[9]:


#------------------用户另一个点出行时间向量的计算----------------
for k in ucomt.keys():
    ucomtvec[k] = {}
    for k1 in ucomt[k].keys():
        ucomt[k][k1]/=np.sqrt(np.sum(ucomt[k][k1]*ucomt[k][k1]))
        ucomtvec[k][k1] = np.zeros((1,64))
        for i in range(4):
            ucomtvec[k][k1] += ucomt[k][k1][i] * tvec[i]
        ucomtvec[k][k1] = torch.tensor(ucomtvec[k][k1]).float()


# # 求用户在不同场景下的向量  目前分为  娱乐 办公 教育三个场景
fakedes = ['其他','商旅','大型枢纽','访友探亲','医疗','生活服务']
realpoint = {'休闲娱乐':1,'办公地点':2,'教育':3}
usrpoi = {}#用户出现的poi场景集合
entvec = np.zeros((len(usrlist),128))#娱乐poi向量（下同）
offvec = np.zeros((len(usrlist),128))#办公
eduvec = np.zeros((len(usrlist),128))#教育
gve = {}
for usr in usrlist:
    recorddata = train_data[usr]
    uid = usr2dig[usr]
    count = np.zeros(4)#保存每个出行场景下的数量
    for record in recorddata:
        if record[6] not in fakedes:
            pp = record[0].split(' ')
            if isnan(record[4]) == False and record[4].split(':')[0] in realpoint.keys():
                    if pp[0] in vocab:
                        g = realpoint[record[4].split(':')[0]]
                        v1 = vecmat[pp[0]][:128] * weight_delay(record[1])  
                        gve[pp[0]] = vecmat[pp[0]][:128]
                    
            if isnan(record[5]) == False and record[5].split(':')[0] in realpoint.keys():
                    if pp[1] in vocab:
                        g = realpoint[record[5].split(':')[0]]
                        v1 = vecmat[pp[1]][:128]  * weight_delay(record[1])
                        gve[pp[1]] = vecmat[pp[1]][:128]
            count[g]+=1
            if g==1:
                entvec[uid] += v1
            if g==2:
                offvec[uid] += v1
            if g==3:
                eduvec[uid] += v1  
    if count[1] != 0:
        entvec[uid] /= np.linalg.norm(entvec[uid])
    if count[2] != 0:
        offvec[uid] /= np.linalg.norm(offvec[uid])
    if count[3] != 0:
        eduvec[uid] /= np.linalg.norm(eduvec[uid])
# # 将点的向量转换为tensor
for key in list(gve.keys()):
    gve[key] = torch.tensor(gve[key]).float()
    

# # 寻找最相似的K个用户，并求得可能的休闲场所（不包括用户历史中存在的）
def usr_sim(usrpoi,topk):
    zerosvec = np.zeros(128*3)#零向量
    uve1 = torch.tensor(usrpoi)
    usr_sim = torch.mm(uve1[:3000,:],uve1.t())
    a1, a2 = usr_sim.sort(1,True)
    a1 = a1[:,1:topk+1]
    a2 = a2[:,1:topk+1]
    i = 3000
    while (1):
        usr_sim = torch.mm(uve1[i:min(i + 3000, len(usrlist)),:], uve1.t())
        a11, a12 = usr_sim.sort(1, True)
        a1 = torch.cat((a1, a11[:,1:topk+1]), 0)
        a2 = torch.cat((a2, a12[:,1:topk+1]), 0)
        print(a1.shape)
        i+=3000
        if i > len(usrlist):
            break
    sim_sortnum = a1.detach().numpy()
    sim_sortid = a2.detach().numpy()
    #如果这个向量本来就是零向量，则他是没有相似的向量
    for i in range(0,usrpoi.shape[0]):
        if (usrpoi[i] == zerosvec).all():
            sim_sortid[i][0] = -1
    return (sim_sortid,sim_sortnum) #（排序后的id，排序后的值）
def point_sim(usrssim ,ssimnum ,udd,uweek,topk):
    def filter_point(goods, usrweight, imp, usr_g):#对于每个不同的娱乐场所，只记录其对距离现在最近的一次,这里去除了在用户历史中出现的数据
        df = {}     #四个参数分别为被选用户的点，被选中用户的权重,这些点的权重，以及目标用户的商品
        i = 0 
        for i in range(len(goods)):
            if goods[i] not in usr_g and goods[i] not in df.keys():
                df[goods[i]] = imp[i] * usrweight
        return df
    def dict_add(df1,df2):#两个字典相加，如果有相同的key，则value相加
        for k in df2.keys():
            if k in df1.keys():
                df1[k] += df2[k]
            else:
                df1[k] = df2[k]
        return df1
    usrg = {}
    for uid in range(usrssim.shape[0]):
        usrg[uid] = []
        if usrssim[uid][0] == -1:
            continue
        df = {}
        for i in range(topk):
            df2 = filter_point(udd[usrssim[uid][i]],ssimnum[uid][i] , uweek[usrssim[uid][i]], udd[uid])
            df = dict_add(df,df2)
        result = sorted(df.items(), key=lambda item:item[1],reverse=True)
        for i in range(3):
            usrg[uid].append(result[i][0])
    return usrg
top_k = 100
(sim_sortid,sim_sortnum) = usr_sim(np.hstack((entvec,offvec,eduvec)),top_k)
usrg4 = point_sim(sim_sortid ,sim_sortnum ,udd ,uweek,top_k)
del sim_sortid
del sim_sortnum


##查到与去过的历史记录中的top2
def counter(arr):
    d = 100
    m = Counter(arr).most_common(d)
    g = []
    for data in m:
        (k,v) = data
        if v > 1:
            g.append(k)
    return g
ud = {}
goods = []
usrve = entvec
for uid in udd.keys():
    usetd = list(set(udd[uid]))
    goods += usetd
    a = np.zeros(len(usetd))
    for i in range(len(usetd)):
        if usetd[i] not in gve.keys():
            continue
        if np.linalg.norm(gve[usetd[i]]) == 0:
            a[i] = 0
        else:
            a[i] = np.dot(gve[usetd[i]],usrve[uid])/(np.linalg.norm(gve[usetd[i]])*(np.linalg.norm(usrve[uid])))
    goodsid = np.argsort(-a)[:min(3, len(usetd))]
    ud[uid] = []
    for id in goodsid:
        ud[uid].append(usetd[id])


# In[10]:



# # 评价代码
need = torch.tensor(range(1,12)).float()
def getweight(start,end,tll,weight):
    vs = torch.zeros(len(start),64)
    ve = torch.zeros(len(start),64)
    vt = torch.zeros(len(start),64)
    a = np.ones(len(end))
    for i in range(len(start)):
        if start[i] not in gve.keys() or end[i] not in gve.keys():
            a[i] = 0
            continue
        vs[i] = gve[start[i]][:64]
        ve[i] = gve[end[i]][64:]
        vt[i] = tll[i]
    (out1, out2) = net(vs,ve,vt)
    re = (F.softmax(out1)[:,1] * torch.sum(F.softmax(out2) * need, 1)).detach().numpy()
    for i in range(len(weight)):
        if a[i] == 0:
            weight[i] = weight[i]
        else:
            weight[i] = weight[i]*re[i]
    return weight

def counter(arr):
    d = 100
    m = Counter(arr).most_common(d)
    g = []
    for data in m:
        (k,v) = data
        if v > 1:
            g.append(k)
    return g
#从列表数据data中按照其对应的权重weight列表随机选取num个数据
def weight_choice(data,weight,num):
    choice = []
    num = min(len(data),num)
    weight = np.array(weight)
    idx = np.argsort( -1 * weight)
    for id in idx[:num]:
        choice.append(data[int(id)]) 
    return choice
#获得要预测的边的集合  sset是休闲娱乐点的集合，eset是另一个点的集合，edgelist是该用户历史边的集合，num是随机选择num个数据
#筛选边的规则说明，预测的边，在图中权重要大于60（一天一单），家既能作为起点又能作为终点，教育跟公司作为起点
def getpreedge(sset,eset,edgelist,num,uid):
    propud = np.array([8.74,4.79,2.21,1.8])/1.8
    propucom = np.array([2.3,1,0.33])
    result = []
    weight = []
    #起点，终点，时间向量
    start = []
    end = []
    tll = []
    for i in range(len(sset)):
        for j in range(len(eset)):
                llist = eset[j].split(' ')
                point = llist[0]
                checkstr = llist[1]           
                if int(sset[i]) in orgraph.keys() and int(point) in orgraph[int(sset[i])].keys():
                    if checkstr =='住宅地名':
                        edge1 = sset[i] + ' ' + point #娱乐点作为起点
                        if edge1 not in edgelist:
                            result.append(edge1)
                            start.append(sset[i])
                            end.append(point)
                            tll.append(ucomtvec[uid][eset[j]])
                            weight.append(propud[i] * propucom[j] * 1 )
                        edge2 = point + ' ' + sset[i] #娱乐点作为终点
                        if edge2 not in edgelist:
                            result.append(edge2)
                            start.append(sset[i])
                            end.append(point)
                            tll.append(ucomtvec[uid][eset[j]])
                            weight.append(propud[i] * propucom[j] * 1.6)   
                    else:
                        edge2 = point+' '+sset[i] #娱乐点作为终点
                        if edge2 not in edgelist:
                            result.append(edge2)
                            start.append(sset[i])
                            end.append(point)
                            tll.append(ucomtvec[uid][eset[j]])
                            weight.append(propud[i] * propucom[j])
    if len(weight)>0:
        weight = getweight(start,end,tll,weight)
    return (weight_choice(result,weight,num),len(result))
sum = 0
m = 0
m1 = 0
i = 0
for u in usrlist:
    record = test_data[u]
    s = record[0].split(' ')[0]
    e = record[0].split(' ')[1]
    uid = usr2dig[u]
    sst = usrg4[uid]
    if u in usr1list:#没有休闲娱乐就全部是协同
        sset = sst
    else:
        if u in usr2list:#有一单，则取这个单来用一下
            sset = ud[uid][:1]+ sst[:3]
        else:#大于等于两单则取两单
            sset = ud[uid][:2] +  sst[:2]
    if (isnan(record[4])==False and record[4].split(':')[0]=='休闲娱乐' and s in sset) or (isnan(record[5])==False and record[5].split(':')[0]=='休闲娱乐' and e in sset):
              eset = ucom[uid][:2]#另一个点的集合
              result = getpreedge(sset,eset,set(uedge[uid]),5,uid)
              if record[0] in result[0]:
                    m+=1
              m1+=1
              sum += result[1]
    if (i+1)%1000 == 0:
        print((i+1), m/(i+1))
    i += 1
print('acc=', m/len(usrlist))