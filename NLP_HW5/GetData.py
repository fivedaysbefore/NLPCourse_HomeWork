import jieba
import os
import re
import numpy as np

# 数据预处理
def load_vectors(fname):
    fin = open(fname, "r",encoding='utf-8')
    w2v = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        w2v[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return w2v

def read_dataset(maxlen=40, step=3):
    print('preparing datasets...')
    path = 'data/dataset/'
    files = os.listdir(path)
    text = '' 
    for file in files:
        if not os.path.isdir(file):
            text += open(os.path.join(path, file), 'r', encoding="ANSI").read().strip()
    text = text[:len(text)//10]
    text = text[100:]

    text = text.replace("\u3000", '')
    text = text.replace(" ", '')
    text_words = jieba.lcut(text)  

    print('total words:', len(text_words))
    sentences = []  
    next_words = []  
    for i in range(0, len(text_words) - maxlen, step):
        sentences.append(" ".join(text_words[i: i + maxlen]))  
        next_words.append(text_words[i + maxlen]) 
    print('nb sentences:', len(sentences))

    print('tokenizing...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
    tokenizer.fit_on_texts(text_words)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    X = tokenizer.texts_to_sequences(sentences)
    y = tokenizer.texts_to_matrix(next_words)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen)
    y = np.array(y)
    print('vocab size:', len(tokenizer.word_index))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)

    if os.path.exists('embedding_matrix.dat'):
        print('building embedding_matrix...')
        embedding_matrix = pickle.load(open('embedding_matrix.dat', 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_vectors('word2Vec.txt')
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        print('building embedding_matrix...')
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
            else:
                embedding_matrix[word_index[word]] = 1
        pickle.dump(embedding_matrix, open('embedding_matrix.dat', 'wb'))

    return tokenizer, index_word, embedding_matrix, text_words, X, y