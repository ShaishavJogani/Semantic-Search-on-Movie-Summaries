#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 23:41:35 2018

@author: siddartha
"""
from nltk.corpus import wordnet as wn
import numpy as np
import data_helpers
import auxilary_data_helper
from tokenizeData import Lemmatizer
import itertools
import pickle
from keras.models import load_model
from neuralcoref import Coref

def evaluate_on_test(x_test, y_test, model):
    predictions = model.predict(x_test)
    evaluate = model.evaluate(x_test, y_test)
    #evaluate = model.evaluate(testStr, np.array([[0,1], [0,1],[0,1], [1, 0]]))
    print("Evaluated against test dataset: " + evaluate)


def predict_movie(testStr, model, labels_dict, vocabulary, event_voc, ners_voc, multiple = False):
    coref = Coref()
    

    clusters = coref.one_shot_coref(utterances= testStr)
    testStr = coref.get_resolved_utterances()
    testStr_vector = transform_testdata([testStr], vocabulary)
    events_onehot = extract_events_onehot([testStr], event_voc)
    
    ners_onehot = extract_ners_onehot([testStr], ners_voc)
    sent_vector = predict_helper.sent_embed(testStr)

    pred = model.predict([testStr_vector, events_onehot, ners_onehot,sent_vector])
    
    sorted_pred = pred[0]
    
    if(not multiple):
        print(labels_dict[sorted_pred.argmax()])
        return
    pred_dict = {i: x for i, x in enumerate(sorted_pred)}
    sorted_pred_dict = sorted(pred_dict, key=pred_dict.get, reverse=True)
    sliced_pred = sorted_pred_dict[:5]
    for r in sliced_pred:
        print(labels_dict[r] + str(pred_dict[r]))

def transform_testdata(test_strs, vocabulary):
    test_strs = [Lemmatizer(data_helpers.clean_str(sent)) for sent in test_strs]
    test_strs = [s.split(" ") for s in test_strs]
    
    test_strs_padded = data_helpers.pad_sentences(test_strs, testStringLength = 90)
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary["<PAD/>"] for word in sentence] for sentence in test_strs_padded])
    return x

 
def extract_events_onehot(testStrs, event_voc):
    events_all_test_str = []
    for testStr in testStrs:
        sentences = testStr.split('.')
        events_in_str = filter(None, auxilary_data_helper.extract_events(sentences))
        events_all_test_str.append(list(set(events_in_str)))
        
    events_all_test_str = auxilary_data_helper.pad_sentences(events_all_test_str, testStringLength=1379)
    synonyms = get_synonyms(events_all_test_str)
    return auxilary_data_helper.build_input_data(events_all_test_str, event_voc, synonyms=synonyms, is_test = True)
    
def extract_ners_onehot(testStrs, ners_voc):
     ners_all_test_str = []
     for testStr in testStrs:
         ners_in_str = auxilary_data_helper.extract_ners(testStr)
         ners_all_test_str.append(ners_in_str)
     ners_all_test_str = auxilary_data_helper.pad_sentences(ners_all_test_str, testStringLength=1379)
     return auxilary_data_helper.build_input_data(ners_all_test_str, ners_voc)

def get_synonyms(events_all_summaries):
    synonyms ={event: get_synonyms_for_word(event) for event in list(itertools.chain.from_iterable(events_all_summaries))}
    return synonyms        
    
def get_synonyms_for_word(word):
    synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def load_model_and_params():
    with open ('model_params', 'rb') as fp:
        model_params = pickle.load(fp)
    model = load_model('model.h5')
    return model, model_params
    
