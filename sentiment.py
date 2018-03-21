# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:09:08 2018

@author: DiveshSoni
"""

from flask import Flask, jsonify, request, render_template
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)


classifier = None

@app.route('/')
def hello():
    global classifier
    neg_reviews = []
    for fileid in movie_reviews.fileids('neg'):
        words = movie_reviews.words(fileid)
        neg_reviews.append((create_word_features(words), "negative"))

    pos_reviews = []
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append((create_word_features(words), "positive"))

    train_set = neg_reviews[:750] + pos_reviews[:750]
    test_set =  neg_reviews[750:] + pos_reviews[750:]
    #print(len(train_set),  len(test_set))

    classifier = NaiveBayesClassifier.train(train_set)


    return 'hello'


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict



@app.route('/result')
def result():
    global classifier
    data1 = open("reviews.txt", "r")
    data = data1.read()

    words = word_tokenize(data)
    words = create_word_features(words)
    file = open("Judgement.txt", "w+")
    return jsonify({'analysis': classifier.classify(words)})


app.run(port=5000)

