import os
import re

class GetData():

    def __init__(self, path):
        self.path = path

    def getCorpus(self):
        corpus = []
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 过滤非法字符
        dirs = os.listdir(self.path)
        count = 0
        for file in dirs:
            filepath  = os.path.join(self.path, file)
            if os.path.isfile(filepath):
                with open(os.path.abspath(filepath), "r", encoding='ANSI') as file:
                    filecontext = file.read()
                    filecontext = re.sub(r1, '', filecontext)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com",'')
                    count += len(filecontext)
                    corpus.append(filecontext)
            elif os.path.isdir(filepath):
                GetData.AllFiles(self, filepath)
        return corpus,count