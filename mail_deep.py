import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, ConvLSTM2D, BatchNormalization, RepeatVector, Conv2D, LSTM, Dropout, Dense
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping


from sklearn.preprocessing import LabelEncoder


import tqdm
import numpy as np
import load_data



EMBEDDING_SIZE = 100
SEQUENCE_LENGTH = 100
TEST_SIZE = 0.5
FILTERS = 70
BATCH_SIZE = 100
EPOCHS = 5

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


def get_model(tokenizer, lstm_units): # builds the lstm model
      
    embedding_matrix = get_embedding_vectors(tokenizer) # loads glove embedding 
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 
              EMBEDDING_SIZE,
              weights=[embedding_matrix],
              trainable=False,
              input_length=SEQUENCE_LENGTH))
   
    model.add(LSTM(lstm_units, recurrent_dropout=0.3))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation="softmax")) #probobility studff 
    

    # rmsprop better than adam 
    #weights[0] = weights[0].reshape(list(reversed(weights[0].shape)))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    return model

# give lables a numeric value
label2int = {"ham": 0, "phishing": 1}
int2label = {0: "ham", 1: "phishing"}

tokenizer = Tokenizer() #converts the utf-8 into tokinized characters 

#Load data
body, subject, label = load_data.load()

tokenizer.fit_on_texts(body)
# tokinize into ints 
X = tokenizer.texts_to_sequences(body)

X = np.array(X)
y = np.array(label)
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [ label2int[label] for label in y ] #loads lables 
y = tf.keras.utils.to_categorical(y)


from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

model = get_model(tokenizer=tokenizer, lstm_units=1024)  # adds LSTM unnits to the model 

# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights poor attempt at better learnrin model 
model_checkpoint = ModelCheckpoint(
            filepath, verbose=0,
            save_best_only=False, save_weights_only=False,
            save_freq="epoch")

# print our data shapes
print("X_train.shape:", X.shape)

print("y_train.shape:", y.shape)

# train the model
history = model.fit(X, y, validation_data=(X, y),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[model_checkpoint],
          verbose=1)



