#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:58:02 2018

@author: siddartha
"""
import predict_helper
import pandas as pd


queries_df = pd.read_csv('./data/test.txt', sep="*", delimiter="*", header = None)
test_queries = queries_df[1].tolist()
test_labels = queries_df[0].tolist()

#predict
model, model_params = predict_helper.load_model_and_params()
test_predictions = predict_helper.predict_movie(test_queries, model, model_params['labels_dict'], model_params['vocabulary'], model_params['event_voc'], model_params['ners_voc'], multiple = True)

for i in range(len(test_labels)):
    label = test_labels[i]
    pred_label = test_predictions[i]
    print("Actual: "+ label+", Predicted: "+pred_label+"\n")