from GetData import GetData
import CalWordBased as cal
# import CalCharBased as cal

if __name__ == '__main__':

#实例化数据类
    FileData = GetData("./data")
#数据处理
    corpus, count = FileData.getCorpus()
#计算
    cal.CalUnigram(corpus, count)
    cal.CalBigram(corpus,count)
    cal.CalTrigram(corpus,count)



