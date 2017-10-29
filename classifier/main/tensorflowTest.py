import tensorflow as tf
import os
import csv
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
import scipy.stats
from dataExtract.dataExtractor import *

"""
# Test file for tensorflow shenanigans
#
#
"""


conEx = ConVoteExtractor()

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
docs = []
values = list(conEx.process().values())
combined = values[0] + values[1]

for i, text in enumerate(combined):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

model = Doc2Vec(size=100, window=10, min_count=5, workers=4, iter=5)
model.build_vocab(docs)
model.train(docs, total_examples=2740, epochs=model.iter)




#model = Doc2Vec(alpha=0.025, min_alpha=0.025)
#model.build_vocab(map (lambda x:(x[1]), entries))

'''os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([0.1], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1,2,3,4]}))
print(sess.run(loss, {x:[1,2,3,4], y:[]}))'''


