#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

""" Twitter Sentiment Analysis
# Ricardo Sousa
# rsousa at rsousa.org

# 2015 Ricardo Sousa
# Part of Sentiment Analysis Project
"""

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def process_data(filein):

    labels = dict(negative=-1, neutral=0, positive=1)
    train_data_x = []
    test_data_x  = []
    train_data_y = []
    test_data_y  = []

    # tokenizer for url {2,6}?$
    urlregexp = RegexpTokenizer('(https?:\/\/)?([\w\.-]+)\.([a-z\.]{2,6})([\/\w]*)*', flags=re.UNICODE)
    words = RegexpTokenizer('\w+', flags=re.UNICODE)

    count = 0
    f = open(filein, mode="r")
    lines = f.readlines()
    train_size = np.round(0.8*len(lines))
    print "Loading..\b",
    for line in lines:
        if count % 100 == 0:
            print "\b.",
        linesplit = line.rstrip().split('\t')

        label = labels[linesplit[1]]

        # sentence
        sentence  = linesplit[2].decode('utf8')
        # pass to lower
        sentence  = sentence.lower()

        #print sentence
        #print type(sentence)
        # remove url's from sentence
        urlexp   = urlregexp.tokenize(sentence)
        #print urlexp
        if len(urlexp) > 0:
            for item in urlexp:
                sentence = string.replace(sentence, item, "")

        sentence = ''.join(sentence)
        # print sentence

        # remove punctuation and returns tokens
        tokens = words.tokenize(sentence)
        #print "PUNCTUATION", sentence

        # steaming
        porter = PorterStemmer()
        tokens = map(lambda x: porter.stem(x), tokens)
        #print tokens

        # remove stop words
        filtered = [w for w in tokens if not w in stopwords.words('english')]

        filtered = ' '.join(filtered)
        if count < train_size:
            train_data_x.append(filtered)
            train_data_y.append(label)
        else:
            test_data_x.append(filtered)
            test_data_y.append(label)
        count += 1

    vectorizer = TfidfVectorizer(min_df=1)
    tidf_train_x = vectorizer.fit_transform(train_data_x)
    tidf_test_x  = vectorizer.transform(test_data_x)

    print "\b.",
    print ".done"
    return tidf_train_x, tidf_test_x, train_data_y, test_data_y

if __name__ == "__main__":
    processdata(None)