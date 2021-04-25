#%%
# encoding=utf-8
import os
import re
import sys
import time
import pandas as pd
import PIL
from PIL import Image
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from opencc import OpenCC
import jieba
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator
import keras.utils
from keras import backend as K
from keras.models import Sequential,Model  #用來啟動 NN
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Dropout,Permute,RepeatVector,Lambda\
,Activation,Input,GlobalAveragePooling1D,GlobalAveragePooling2D,multiply,Multiply,Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Flatten, Dense, LSTM,CuDNNGRU,CuDNNLSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
maxlen = 300
max_words = 30000
#%%
df = pd.read_excel('data.xlsx')
df['text'] = df.comment.apply(lambda x: " ".join(jieba.cut(str(x))))
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.text)

#%%
sequences = tokenizer.texts_to_sequences(df.text)
data = pad_sequences(sequences, maxlen=maxlen)
word_index = tokenizer.word_index
labels = np.array(df.sentiment)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

#%%
data = data[indices]
labels = labels[indices]
training_samples = int(len(indices) * 0.8)
validation_samples = len(indices) - training_samples

#%%
X_train = data[:training_samples]
y_train = labels[:training_samples]
X_valid = data[training_samples: training_samples + validation_samples]
y_valid = labels[training_samples: training_samples + validation_samples]
X_all = data[:]
y_all = labels[:]
del training_samples
del validation_samples
#%%
zh_model = KeyedVectors.load_word2vec_format('cna.cbow.512d.0.txt')
zh_model.vectors[0]
embedding_dim=len(zh_model[next(iter(zh_model.vocab))])
embedding_matrix = np.random.rand(max_words, embedding_dim)
embedding_matrix = (embedding_matrix - 0.5) * 2 
#%%
for word, i in word_index.items():
    if i < max_words:
        try:
          embedding_vector = zh_model.get_vector(word)
          embedding_matrix[i] = embedding_vector
        except:
          pass 

#%%
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    attention = Dense(1, activation='tanh')(inputs)                             # input shape = batch * time_steps * 1
    attention = Flatten()(attention)                                            # input shape = batch * time_steps
    attention = Activation('softmax')(attention)                                # input shape = batch * time_steps
    attention = RepeatVector(input_dim)(attention)                              # input shape = batch * input_dim * time_steps
    attention = Permute([2, 1])(attention)                                      # input shape = batch * time_step * input_dim
    sent_representation = multiply([inputs, attention])                         # input shape = batch * time_step * input_dim
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),               # input shape = batch * input_dim
                                 output_shape=(input_dim,))(sent_representation)
    return sent_representation
#%%
inputs = Input(shape=(X_train.shape[1],))

Embedding0_layer=Embedding(max_words, embedding_dim)(inputs)

LSTM_layer = Bidirectional(LSTM(512,recurrent_dropout=0.2,dropout=0.2,return_sequences=True))(Embedding0_layer)
LSTM_layer_1 = Bidirectional(LSTM(256,recurrent_dropout=0.2,dropout=0.2,return_sequences=False))(LSTM_layer)

attout = attention_3d_block(Embedding0_layer)
attention_mul = multiply([LSTM_layer_1,attout])

dense0_layer = Dense(256, activation='sigmoid')(attention_mul)
DP0_layer = Dropout(0.2)(dense0_layer)
dense1_layer = Dense(128, activation='sigmoid')(DP0_layer)
DP1_layer = Dropout(0.2)(dense1_layer)

output = Dense(1, activation='sigmoid')(DP1_layer)
model = Model(input=[inputs], output=output)
model.summary()

#%%
model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False
#%%
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
EarlyStopp = EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="auto")
ModelCheckpoint0 = ModelCheckpoint('save_model/my_model.h5',monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,mode='auto', period=1)
history = model.fit(X_all, y_all,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    ### You just comment out this code (validation_data...) , It's works after uncommenting it
                    #validation_data=(X_valid, y_valid),
                    callbacks=[ModelCheckpoint0])
#%%訓練結果可視化
plt.plot(history.history['val_acc'],'r-')
plt.plot(history.history['val_loss'],'g-')

plt.plot(history.history['acc'],'r--')
plt.plot(history.history['loss'],'g--')
#%%
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
model.save('save_model/model.h5')
from keras.utils import plot_model
plot_model(model, to_file='model.png')
#%%

# 驗證模型
score = model.evaluate(X_valid, y_valid, verbose=0)
#X_val_test, y_val_test
# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])
