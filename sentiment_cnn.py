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
import itertools
from tokenizeData import Lemmatizer


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
    
    events_onehot, event_voc, event_voc_inv_list, events_labels_dict = EventDataHelper.load_event_data()
    event_voc_inv = {key: value for key, value in enumerate(event_voc_inv_list)}
    x2 = transform_events_input(events_onehot, labels_y, labels_dict)
    
    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    x2 = x2[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    x2_train = x2[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    x2_test = x2[train_len:]
    y_test = y[train_len:]

    return x_train,  x2_train, y_train, x_test, x2_test,  y_test, vocabulary, vocabulary_inv, event_voc, event_voc_inv, num_labels, labels_dict

def evaluate_on_test(x_test, y_test):
    predictions = model.predict(x_test)
    evaluate = model.evaluate(x_test, y_test)
    #evaluate = model.evaluate(testStr, np.array([[0,1], [0,1],[0,1], [1, 0]]))
    print("Evaluated against test dataset: " + evaluate)
    
def extract_events_onehot(testStrs, event_voc):
    events_all_test_str = []
    for testStr in testStrs:
        sentences = testStr.split('.')
        events_in_str = filter(None, EventDataHelper.extract_events(sentences))
        events_all_test_str.append(list(set(events_in_str)))
        
    events_all_test_str = EventDataHelper.pad_event_sentences(events_all_test_str, testStringLength=1379)
    return EventDataHelper.build_input_data(events_all_test_str, event_voc)
    
def predict_movie(testStr, model, labels_dict, event_voc, multiple = False):
    testStr_vector = transform_testdata([testStr])
    events_onehot = extract_events_onehot([testStr], event_voc)
    pred = model.predict([testStr_vector, events_onehot])
    
    sorted_pred = pred[0]
    
    if(not multiple):
        print(labels_dict[sorted_pred.argmax()])
        return
    pred_dict = {i: x for i, x in enumerate(sorted_pred)}
    sorted_pred_dict = sorted(pred_dict, key=pred_dict.get, reverse=True)
    sliced_pred = sorted_pred_dict[:5]
    for r in sliced_pred:
        print(labels_dict[r])

def transform_testdata(test_strs):
    test_strs = [Lemmatizer(data_helpers.clean_str(sent)) for sent in test_strs]
    test_strs = [s.split(" ") for s in test_strs]
    
    test_strs_padded = data_helpers.pad_sentences(test_strs, testStringLength = 90)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in test_strs_padded])
    return x

    
    
# Data Preparation
print("Load data...")
x_train,  x2_train, y_train, x_test, x2_test,  y_test, vocabulary, vocabulary_inv, event_voc, event_voc_inv, num_labels, labels_dict = load_data()

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size: {:d}".format(x_test.shape[1]))
    sequence_length = x_test.shape[1]

if events_seq_length != x2_test.shape[1]:
    print("Adjusting event sequence length for actual size: {:d}".format(x2_test.shape[1]))
    events_seq_length = x2_test.shape[1]
    
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))


# Prepare embedding layer weights and convert inputs for static model
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)

# Build model
input_shape = (sequence_length,)
model_input = Input(shape=input_shape, name="Input_Layer")
embed = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding_layer")(model_input)
embed = Dropout(dropout_prob[0], name="embedding_dropout_layer")(embed)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1, name="conv_layer_"+ str(sz))(embed)
    conv = MaxPooling1D(pool_size=2, name="conv_maxpool_layer_"+ str(sz))(conv)
    conv = Flatten(name="conv_flatten_layer_"+ str(sz))(conv)
    conv_blocks.append(conv)
    
conv_layers = Concatenate(name="conv_concate_layer")(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
flated_conv_layers = Dropout(dropout_prob[1], name="concate_dropout_layer")(conv_layers)

events_input_layer = Input(shape=(events_seq_length,), name="event_input_layer")
events_dense = Dense(int((events_seq_length/2)), activation="relu", name="event_dense_layer")(events_input_layer)
#flatted_events = Flatten(name="event_flatten_layer")(events_dense)

merged = concatenate([flated_conv_layers, events_dense], name="conv_event_merge_layer")

dense = Dense(hidden_dims, activation="relu", name="conv_event_merge_dense_layer")(merged)
model_output = Dense(num_labels, activation="softmax", name="Output_layer")(dense)

model = Model([model_input, events_input_layer], model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Model Summary
print(model.summary())
plot_model(model, to_file='event_summary_classification.png')

# Initialize weights with word2vec
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding_layer")
embedding_layer.set_weights([weights])

# Train the model
model.fit([x_train, x2_train], y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=([x_test, x2_test], y_test), verbose=2)


model.save("model.h5")

predict_movie("Hero is a lawyer or attorney in a small-town in Michigan. Bids for a reelection and loses.", model, labels_dict, event_voc, multiple = True)

