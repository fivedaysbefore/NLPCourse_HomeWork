# -*- coding: utf-8 -*-
from gensim import  models
from sklearn import cluster
from GetData import get_data

# 运用Word2Vec进行训练
if __name__ == "__main__":
    get_data() 
    fr = open('./train_data.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)

    num_features = 300  
    min_word_count = 10  
    num_workers = 16  
    context = 10  
    downsampling = 1e-3  
    sentences = models.word2vec.Text8Corpus("./train_data.txt")

    model = models.word2vec.Word2Vec(sentences, workers=num_workers,vector_size=num_features, min_count=min_word_count,window=context, sg=1, sample=downsampling)

# Kmeans 聚类
    names=[]
    for line in open("./金庸小说人名.txt","r",encoding='utf-8'):
        line = line.strip('\n')
        names.append(line)
    names = [name for name in names if name in model.wv]
    name_vectors = [model.wv[name] for name in names]
    n=16
    label = cluster.KMeans(n).fit(name_vectors).labels_
    print(label)
    for i in range(n):
        print("\n类别" + str(i+1) + ":")
        for j in range(len(label)):
            if label[j] == i:
                print(names[j],end=" ")


    print('\n')
    test_male = ['狄云', '杨过', '张无忌', '郭靖', '萧峰']
    test_female = ['小昭','周芷若', '小龙女', '王语嫣', '黄蓉']
    for i in range(5):
        print("与" + str(test_male[i]) + "关系相近的10个词汇:\n" + str(model.wv.most_similar(test_male[i], topn=10)))
    for i in range(5):
        print("与" + str(test_female[i]) + "关系相近的10个词汇:\n" + str(model.wv.most_similar(test_female[i], topn=10)))   