
## Note that '299' was used in this example to illustrate that one can choose any number of steps or sequences for the LSTM unit to process -- ie. there is no special consideration for 'x' in LSTM(x) when constructing the topology of the Deep Neural Network.


import numpy
import numpy as np
import pandas as pd
import keras.preprocessing.text
import string
import re
from pandas import read_csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, GRU, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# Set random seed for reduced variation
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(77)

datafile = 'data/tweets.csv'
#datafile2 = 'data/tweets2a.csv'  ## Not using the "Output to 'datafile2' and read in back from 'datafile2' later approach
#cols = ['tweet','sentiment']
cols = ['tweet']

dataframe = pd.read_csv(datafile, dtype=str, names=cols, header=None)
#dataframe.columns = ["tweet"]
#dataset = dataframe.values

word_index = imdb.get_word_index(path='imdb_word_index.json')
#print word_index

# Translate input tweets to word indexes (correlated to the pre-processed word indexes from the Keras IMDB dataset)
listoflists = []
list = []

#f = open(datafile2, mode="w")  ## Not using the "Output to 'datafile2' and read in back from 'datafile2' later approach 
for row_id, row in enumerate(dataframe.values):
   row = str(row)
   words = re.sub('['+string.punctuation+']', '', row).split()
   words2 = np.array([word_index[word] if word in word_index else 0 for word in words])
   list = words2.tolist()
   listoflists.append(list)
#   f.write(str(row_id) + "," + str(words2) + "\n")  ## Not using the "Output to 'datafile2' and read in back from 'datafile2' later approach 
#   f.write(str(words2) + "\n")  ## Not using the "Output to 'datafile2' and read in back from 'datafile2' later approach 
#f.close()  ## Not using the "Output to 'datafile2' and read in back from 'datafile2' later approach 

# Load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)  ## In this case, we are setting the vocabulary size to 5000

# Pad dataset to a maximum review length in words
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)  ## In this case, we are setting the maximum review length of the Training dataset to 500
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)  ## In this case, we are setting the maximum review length of the Test dataset to 500
X_pred = sequence.pad_sequences(listoflists, maxlen=max_review_length)  ## In this case, we are setting the maximum review length of the Prediction dataset to 500

# Create the Deep Learning model -- without the last fully-connected dense MLP layer
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(199))
#model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit/train the model
epochs=3
batch_size = 64
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Predictions using the model
predictions = model.predict(X_pred)
predictions_prob = model.predict_proba(X_pred)
predictions_class = model.predict_classes(X_pred)
print("Predictions:")
print(predictions)
print("Predictions' Probabilities:")
print(predictions_prob)
print("Predictions' Classes:")
print(predictions_class)

## Output predictions to a file
counter = 0
f = open("predictions_sentiment_cnn_lstm.csv", mode="w")
for item in predictions:
	counter += 1
	f.write(str(counter) + "," + str(item) + "\n")
f.close()

# Create the Deep Learning model -- -- with the last fully-connected dense MLP layer
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(199))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit/train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Create the Deep Learning model -- Reduce the number of LSTM memory cells to 149
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(149))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit/train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Create the Deep Learning model -- Bi-Directional LSTMs
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(149)))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit/train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Create the Deep Learning model -- GRUs
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(GRU(149)))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit/train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
