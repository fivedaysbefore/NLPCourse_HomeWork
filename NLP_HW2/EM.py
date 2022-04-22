
import math
import numpy as np

def GetData(s1,s2,p,q,r,N):

    #随机生成硬币序列 0代表反面，1代表正面

    CoinSeq = np.zeros(N)
    CoinChance = [p,q,r] 
    for i in range(N):
        #抽取硬币
        rand1 = np.random.random()
        if rand1 < s1:
            coin = 0
        elif rand1 <s1+s2:
            coin = 1
        else:
            coin = 2
        #掷硬币
        rand2 = np.random.random()
        if rand2 < CoinChance[coin]:
            CoinSeq[i] = 1
    return CoinSeq

#隐参数计算
def EStep(s1, s2, p, q, r, coinseq):
    Z1, Z2, Z3 = [], [], []
    for x in coinseq:
        Miu1 = s1*pow(p, x)*pow(1-p,1-x)
        Miu2 = s2*pow(q, x)*pow(1-q,1-x)
        Miu3 = (1 - s1 -s2)*pow(r, x)*pow(1-r, 1-x)
        sum = Miu1 + Miu2 + Miu3
        Miu1 = Miu1/sum
        Miu2 = Miu2/sum
        Miu3 = Miu3/sum
        Z1.append(Miu1)
        Z2.append(Miu2)
        Z3.append(Miu3)
    return [Z1, Z2, Z3]

def MStep(z, coinseq):
    s,p = [],[]
    for i in range(len(z)):
        sum1 = 0.0
        if i < 2:
            s.append(sum(z[i])/len(z[i]))
        for j,x in enumerate(coinseq):
            sum1 += z[i][j]*x
        p.append(sum1/sum(z[i]))
    return s,p

def EM(s, p, coinseq, iterNum, epsilon):
    s_pre, p_pre = s, p
    for i in range(iterNum):
        z = EStep(s_pre[0],s_pre[1],p_pre[0],p_pre[1],p_pre[2],coinseq)
        s_temp , p_temp = MStep(z,coinseq)
        sum1, sum2 = 0.0, 0.0
        for j in range(len(s_temp)):
            sum1 += abs(s_temp[j] -  s_pre[j])
        for k in range(len(p_temp)):
            sum2 += abs(p_temp[k] -  p_pre[k])
        if (sum1 + sum2 < epsilon):
            print("迭代次数：", i+1)
            break
        s_pre, p_pre = s_temp, p_temp

    return  s_pre, p_pre 


if __name__ == '__main__':
    #参数预设
    s1, s2, p, q, r, N = 0.3, 0.2, 0.8, 0.6, 0.7, 10000
    coinseq = GetData(s1,s2,p,q,r,N)
    print(coinseq)
    start_s = [0.4,0.2]
    start_p = [0.4, 0.5, 0.6]

    s_p,p_p = EM(start_s, start_p, coinseq, 10000, 0.0000001)
    print("真实值：s1 = {:.4f}, s2 = {:.4f}, p = {:.4f}, q = {:.4f},  r = {:.4f},".format(s1, s2, p, q, r))
    print("初始值：s1 = {:.4f}, s2 = {:.4f}, p = {:.4f}, q = {:.4f},  r = {:.4f},".format(start_s[0],start_s[1], start_p[0], start_p[1], start_p[2]))
    print("预测值：s1 = {:.4f}, s2 = {:.4f}, p = {:.4f}, q = {:.4f},  r = {:.4f},".format(s_p[0],s_p[1], p_p[0], p_p[1], p_p[2]))
    


