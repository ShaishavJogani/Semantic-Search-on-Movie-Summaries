#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:56:17 2018

@author: siddartha
"""

#import jnius_config

#jnius_config.set_options('-Xms128m', '-Xmx512m')
import os
os.environ['CLASSPATH'] = "/Users/siddartha/Documents/Siddartha/ASU_Courses/Natural Language Processing/K-parser/kparser/kparser.jar:/Users/siddartha/Documents/Siddartha/ASU_Courses/Natural Language Processing/K-parser/Extractor/EventExtractor.jar"

from jnius import autoclass
from tokenizeData import Lemmatizer
import numpy as np
import re
import itertools
from collections import Counter
import io
import pandas as pd
import pickle
import os.path

EventExtractor = autoclass('EventExtraction.EventExtraction')

def extract_events(sentences):
    event_extractor = EventExtractor()
    extracted_sentences = event_extractor.eventExtractionEngine(sentences)

    length = extracted_sentences.size()  
    events = []
    for i in range(length):
        tags = extracted_sentences.get(i).split(",")
        event = Lemmatizer(tags[1].split('-')[0])
        events.append(event)
    return events
        
        
def load_event_data():
    events_all_summaries= []
    
    summaries, labels, num_labels, actual_labels = load_event_data_and_labels()
    
    labels_temp = range(num_labels)
    labels_dict = zip(actual_labels, labels_temp)
    labels_dict = set(labels_dict)
    labels_dict = {x[1]: x[0] for i, x in enumerate(labels_dict)}

    
    
    if(os.path.exists('./events')):
        print("Reading from existing events")
        with open ('events', 'rb') as fp:
            events_all_summaries = pickle.load(fp)
    else:
        for i in range(len(summaries)):
            summary = summaries[i]
            sentences = summary.split('.')
            events_in_summary = filter(None, extract_events(sentences))
            events_all_summaries.append(list(set(events_in_summary)))
        with open('events', 'wb') as fp:
            pickle.dump(events_all_summaries, fp)
            
    events_all_summaries = pad_event_sentences(events_all_summaries)
    vocabulary, vocabulary_inv = build_event_vocab(events_all_summaries)
    events_onehot = build_input_data(events_all_summaries, vocabulary)
    
    return [events_onehot, vocabulary, vocabulary_inv, labels_dict]

      

def build_input_data(events_all_summaries, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in event_summary] for event_summary in events_all_summaries])
    num_events = len(vocabulary)
    events_onehot_list = np.zeros((len(events_all_summaries), num_events), int)
    for i in range(len(events_onehot_list)):
        one = events_onehot_list[i]
        x1 = x[i]
        one[x1] = 1
    
    return events_onehot_list


def build_event_vocab(events_all_summaries):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*events_all_summaries))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]
      
    
def pad_event_sentences(events_all_summaries, padding_word="<PAD/>", testStringLength = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in events_all_summaries)
    if testStringLength is not None:
        sequence_length = testStringLength
    padded_events = []
    for i in range(len(events_all_summaries)):
        event_summary = events_all_summaries[i]
        num_padding = sequence_length - len(event_summary)
        new_event_summary = event_summary + [padding_word] * num_padding
        padded_events.append(new_event_summary)
    return padded_events
  
    

def load_event_data_and_labels():
    
    df = pd.read_csv("data/trainMovie.csv")
    selected = ['sentiment', 'review']
    labels = sorted(list(set(df[selected[0]].tolist())))
    
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
    x_raw = df[selected[1]].apply(lambda x: x).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    
    
    return [x_raw, y_raw, num_labels, labels]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
