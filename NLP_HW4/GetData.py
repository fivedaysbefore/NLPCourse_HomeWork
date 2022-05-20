import jieba
import os
import re
import numpy as np

# 数据预处理
def get_data(): # 如果文档还没分词，就进行分词
    train_data = "./train_data.txt"
    test_data = "./test_data.txt"
    if not os.path.exists('./train_data.txt'):
        outputs_train = open(train_data, 'w', encoding='utf-8')
        outputs_test = open(test_data, 'w', encoding='utf-8')
        datasets_root = "./data"
        catalog = "inf.txt"

        test_num = 10
        test_length = 20
        with open(os.path.join(datasets_root, catalog), "r", encoding='utf-8') as f:
            all_files = f.readline().split(",")
            print(all_files)

        for name in all_files:
            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='utf-8') as f:
                file_read = f.readlines()
                train_num = len(file_read) - test_num
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                train_text = ""
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                    outputs_train.write(line_seg.strip() + '\n')


                for test in choice_index[train_num:test_num + train_num]:
                    if test + test_length >= len(file_read):
                        continue
                    test_line = ""
                    line = file_read[test]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    for term in seg_list:
                        test_line += term + " "
                    outputs_test.write(test_line.strip()+'\n')

        outputs_train.close()
        outputs_test.close()
        print("数据处理完成")
