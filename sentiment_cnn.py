"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
import data_helpers
from w2v import train_word2vec
import EventDataHelper
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.merge import concatenate
from keras.utils import plot_model


np.random.seed(0)

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 3

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10
events_seq_length = 300


# ---------------------- Parameters end -----------------------
def transform_events_input(event_all_summaries, labels_y, labels_dict):
    new_events_all_summaries = np.array([event_all_summaries[label] for label in labels_y])
    return new_events_all_summaries

def load_data():
    x, y, vocabulary, vocabulary_inv_list, num_labels, labels_dict = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    labels_y = y.argmax(axis=1)
    
    #event_all_summaries, event_voc, event_voc_inv_list = EventDataHelper.load_event_data()
    #event_voc_inv = {key: value for key, value in enumerate(event_voc_inv_list)}
    #x2 = transform_events_input(event_all_summaries, labels_y, labels_dict)
    
    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    #x2 = x2[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    #x2_train = x2[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    #x2_test = x2[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv, num_labels, labels_dict

def evaluate_on_test(x_test, y_test):
    predictions = model.predict(x_test)
    evaluate = model.evaluate(x_test, y_test)
    #evaluate = model.evaluate(testStr, np.array([[0,1], [0,1],[0,1], [1, 0]]))
    print("Evaluated against test dataset: " + evaluate)
    
def predict_movie(testStr, model, labels_dict):
    testStr = transform_testdata([testStr])
    pred = model.predict(testStr)
    
    for a in pred:
        print(labels_dict[a.argmax()])

def transform_testdata(test_strs):
    test_strs = [data_helpers.clean_str(sent) for sent in test_strs]
    test_strs = [s.split(" ") for s in test_strs]
    
    test_strs_padded = data_helpers.pad_sentences(test_strs, testStringLength = 100)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in test_strs_padded])
    return x


# Data Preparation
print("Load data...")
x_train,  y_train, x_test,  y_test, vocabulary, vocabulary_inv, num_labels, labels_dict = load_data()

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]
    
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))


# Prepare embedding layer weights and convert inputs for static model
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)

# Build model
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)
embed = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
embed = Dropout(dropout_prob[0])(embed)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(embed)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
    
conv_layers = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
flated_conv_layers = Dropout(dropout_prob[1])(conv_layers)

event_input_layer = Input(shape=(events_seq_length,))
merged = concatenate([flated_conv_layers, event_input_layer])

dense = Dense(hidden_dims, activation="relu")(flated_conv_layers)
model_output = Dense(num_labels, activation="softmax")(dense)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Model Summary
print(model.summary())
plot_model(model, to_file='event_summary_classification.png')

#get event_train vectors
event_x_train, event_x_test = get_event_data()

# Initialize weights with word2vec
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)



predict_movie("photographer friend Farhan", model, labels_dict)

