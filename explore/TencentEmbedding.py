
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Conv1D, MaxPooling1D, Dense, Flatten, Lambda, LSTM
from keras import backend as K
import tensorflow as tf
import matplotlib as plt
import pandas as pd
import numpy as np
import jieba

jieba.load_userdict("custom_dict.txt")


# In[2]:


def divide_word(df,column='Comment'):
    seg_list = jieba.cut(df[column], cut_all=False)
    return " ".join(seg_list)

data = pd.read_csv('DMSC.csv')
data = data[['Comment', 'Star']].dropna().copy()

data['Comment'] = data.apply(divide_word,axis = 1)

texts = data['Comment'].values
labels = data['Star'].values
labels_index = list(np.sort(data['Star'].unique()))

print('Found %s texts.' % len(texts))


# In[3]:


MAX_SEQUENCE_LENGTH = 80 #样本最长369，可以测试更长的数据
VALIDATION_SPLIT = 0.05 #数据量很大，验证集可以不用20%
EMBEDDING_DIM = 200 #腾讯的维度，不能修改
EPOCHS = 1
BATCH_SIZE = 512

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# In[4]:


embeddings_index = {}
with open('70000-small.txt','r') as f:
    for i,line in enumerate(f): 
        if i == 0:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))


# In[5]:


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[6]:


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[10]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(128, dropout=0.5)(embedded_sequences)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index)+1, activation='softmax')(x)

model = Model(sequence_input, preds)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

