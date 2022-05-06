from DataProcess import ReadData,GetData
from LDA import LDA

if __name__ == '__main__':

    content = ReadData('data')
    print("Data Initialization Successes")
    
    train_data, train_label,test_data, test_label = GetData(content)
    print("Data Division Successes")

    LDA(train_data, train_label,test_data, test_label, 1000)
    print("Finished")