from __future__ import print_function
import jieba
import argparse
import numpy as np
import random
import sys
import os
import pickle
import re
from GetData import load_vectors,read_dataset
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Activation, LSTM, Embedding
EMBEDDING_DIM = 300



class TextGen:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 20
        self.STEP = 3
        self.ITERATION = 500
        self.tokenizer, self.index_word, self.embedding_matrix, self.text_words, self.X, self.y = \
            read_dataset(maxlen=self.MAX_SEQUENCE_LENGTH, step=self.STEP)

        if os.path.exists('saved_model.h5'):
            print('loading saved model...')
            self.model = tf.keras.models.load_model('saved_model.h5')
        else:
            print('Build model...')
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
            print("len:",len(self.tokenizer.word_index))
            x = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                          output_dim=EMBEDDING_DIM,
                          input_length=self.MAX_SEQUENCE_LENGTH,
                          weights=[self.embedding_matrix],
                          trainable=False)(inputs)

            # x = Bidirectional(LSTM(600, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(x)
            # x = LSTM(600, dropout=0.2, recurrent_dropout=0.1)(x)
            x = LSTM(600, dropout=0.2, recurrent_dropout=0.1)(x)
            x = Dense(len(self.tokenizer.word_index) + 1)(x)
            predictions = Activation('softmax')(x)
            model = tf.keras.models.Model(inputs, predictions)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            # plot_model(model, to_file='model.png')
            self.model = model

    @staticmethod
    def sample(preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def train(self):
        """"
        Training model and inference
        """
        # tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        for i in range(1, self.ITERATION):
            print()
            print('-' * 50)
            print('Iteration', i)
            self.model.fit(self.X, self.y, batch_size=128)  # 训练

            self.model.save('saved_model.h5')
            print('model saved')
            self.inference()

    def inference(self, seed=None):
        if seed is None:
            start_index = random.randint(0, len(self.text_words) - self.MAX_SEQUENCE_LENGTH - 1)
            seed_words = self.text_words[start_index: start_index + self.MAX_SEQUENCE_LENGTH]
            seed = "".join(seed_words)
        else:
            seed_words = jieba.lcut(seed)

        for diversity in [0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            print('----- Generating with seed: "' + seed + '"')
            generated = seed
            sys.stdout.write(generated)  

            x = self.tokenizer.texts_to_sequences([" ".join(seed_words)])
            x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.MAX_SEQUENCE_LENGTH)
            for i in range(100):  
                preds = self.model.predict(x, verbose=0)[0]  
                next_index = self.sample(preds, diversity)  
                next_word = self.index_word[next_index]  

                generated += next_word
                x = np.delete(x, 0, -1)
                x = np.append(x, [[next_index]], axis=1)  

                sys.stdout.write(next_word)  
                sys.stdout.flush() 
            print()

if __name__ == '__main__':
    
    TextMod = TextGen()
    TextMod.train()
    TextMod.inference(seed="忽听得那少女口中嘘嘘嘘的吹了几声。白影闪动，") 