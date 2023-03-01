import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, GRU,Bidirectional, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tqdm
import numpy as np
from preprocess import load_data, load_sample
import pdb


EMBEDDING_SIZE = 100
SEQUENCE_LENGTH = 100

# give lables a numeric value
label2int = {"normal": 0, "phishing": 1}
int2label = {0: "normal", 1: "phishing"}

def get_embedding_vectors(tokenizer, dim=EMBEDDING_SIZE):
    embedding_index = {}
    with open("embed/glove.6B.100d.txt", 'r',encoding='utf8',errors = 'ignore') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

def get_model(tokenizer, embedding_matrix, rnn_cell): # builds the lstm model
      
    # embedding_matrix = get_embedding_vectors(tokenizer) # loads glove embedding 
    rnn_units=1024
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 
              EMBEDDING_SIZE,
              weights=[embedding_matrix],
              trainable=False,
              input_length=SEQUENCE_LENGTH))
    if rnn_cell == 'gru':
        model.add (GRU(rnn_units, recurrent_dropout=0.3))
    elif rnn_cell == 'lstm':
        model.add(LSTM(rnn_units, recurrent_dropout=0.3))
    else:
        model.add(Bidirectional(LSTM(rnn_units, recurrent_dropout=0.3), merge_mode = "concat"))

    model.add (Dense(64))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation="softmax")) #probobility studff 
    
    return model

def get_sample_data (sample_path, tokenizer):
    #Load data
    body, subject = load_sample(sample_path)
    # pdb.set_trace()
    body = [body]
    subject = [subject]

    tokenizer.fit_on_texts(body)
    # tokinize into ints 
    X = tokenizer.texts_to_sequences(body)

    X = np.array(X)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

    return X

def get_train_data (train_data_dir, val_size=0.1):

    tokenizer = Tokenizer() #converts the utf-8 into tokinized characters 
    
    #Load data
    body, subject, label = load_data(train_data_dir)

    tokenizer.fit_on_texts(body)
    #Return embeding matrix
    embedding_matrix = get_embedding_vectors(tokenizer)
    # tokinize into ints 
    X = tokenizer.texts_to_sequences(body)

    X = np.array(X)
    y = np.array(label)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)
    y = [ label2int[label] for label in y ] #loads lables 
    y = tf.keras.utils.to_categorical(y)
    
    X, y = shuffle(X, y, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=val_size , random_state=42)
    return X_train, X_val, Y_train, Y_val, tokenizer, embedding_matrix

def get_test_data (test_data_dir, tokenizer):
    
    body, subject, label = load_data(test_data_dir)
    # pdb.set_trace()
    # tokenizer.fit_on_texts(body)
    # tokinize into ints 
    X = tokenizer.texts_to_sequences(body)

    X = np.array(X)
    y = np.array(label)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)
    y = [ label2int[label] for label in y ] #loads lables 
    y = tf.keras.utils.to_categorical(y)
    
    X, y = shuffle(X, y, random_state=0)
   
    return X, y