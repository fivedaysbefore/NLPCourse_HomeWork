import os
import jieba
import random


def ReadData(path):
    content = []
    names = os.listdir(path)
    FileNum = len(names)
    ParaLenth = 1000 #每段长度 
    ParaNum = 200
    SelectNum = (ParaNum//FileNum) + 1
    for name in names:
        NovelName = path + '\\' + name 
        with open(NovelName, 'r', encoding= 'ANSI') as f:
            con = f.read()
            con = DealContent(con)
            con = jieba.lcut(con)  #分词
            #一共16本小说，抽取200个段落
            selectPos = len(con)//SelectNum 
            for i in range(SelectNum):
                SelectStart =  random.randint(selectPos*i, selectPos*(i+1))
                para = con[SelectStart:SelectStart+ParaLenth]
                content.append((name,para))
        f.close()
    content = content[:ParaNum]
    return content

def DealContent(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content

def GetData(content):
    train_data, train_label = [], []
    test_data, test_label = [], []

    random.shuffle(content)
    for i in range(int(len(content)*0.8)): #80%的数据做训练集
        train_label.append(content[i][0])
        train_data.append(content[i][1])

    for i in range(int(len(content)*0.8), int(len(content))):  #剩下20%做测试集
        test_label.append(content[i][0])
        test_data.append(content[i][1])

    return train_data, train_label,test_data, test_label

