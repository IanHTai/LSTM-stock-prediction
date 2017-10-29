import scipy.stats
import numpy as np
from dataExtract.dataExtractor import *
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple, OrderedDict
from classifier.sentiment.classifiers import Classifier
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(Classifier):
    def __init__(self):
        pass
    def summarize(self, separated_dataset):
        summaries = {}
        for classValue, instances in separated_dataset.items():
            summaries[classValue] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*instances)]
        return summaries

    def probability(self, x, mean, stdev):
        if(stdev == 0):
            if(x == mean):
                return 1
            else:
                return 0
        else:
            return scipy.stats.norm.pdf(x, loc=mean, scale=stdev)

    def calcClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.probability(x, mean, stdev)
        return probabilities

    def getTrainInputVectors(self):
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\training_set')

        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        docs = []
        values = list(conEx.process().values())
        combined = values[0] + values[1]

        for i, text in enumerate(combined):
            words = text.lower().split()
            tags = [i]
            docs.append(analyzedDocument(words, tags))

        self.model = Doc2Vec(size=100, window=10, min_count=5, workers=4, iter=5)
        self.model.build_vocab(docs)
        self.model.train(docs, total_examples=2740, epochs=self.model.iter)

        len1 = len(values[0])
        len2 = len(values[1])

        sepVecs = {0:[], 1:[]}
        for i in range(0, len1 + len2):
            if i < len(values[0]):
                sepVecs[0].append(self.model.docvecs[i])
            else:
                sepVecs[1].append(self.model.docvecs[i])
        return sepVecs

    def giveProbs(self, summaries, sentence):
        return self.calcClassProbabilities(summaries, self.model.infer_vector(sentence))

    def train(self):
        self.summaries = self.summarize(self.getTrainInputVectors())

    def test(self, sentence, y):
        print(self.giveProbs(self.summaries, sentence))

nb = NaiveBayes()
nb.train()
print(nb.calcClassProbabilities(nb.summaries, nb.model.docvecs[50]))
print(nb.calcClassProbabilities(nb.summaries, nb.model.docvecs[2000]))

gnb = GaussianNB()
data = nb.model.docvecs
newOnes = np.zeros(len(data))
temp = nb.getTrainInputVectors()
for i in range(len(temp[0]), len(temp[1])):
    newOnes[i] = 1

gnb.fit(data, newOnes)
print(gnb.predict([data[0]]), gnb.predict_proba([data[0]]))
print('\n\n')
print('News Twitter: ', gnb.predict_proba([nb.model.infer_vector('The European Parliament has warned that " hostile propaganda " by Russia against the EU is growing'.lower())]))
dataset = [
    'i have a big cat .',
    'my cat is very big .',
    'dogs are quite nice , i enjoy spending time with them !'
]

def BoWStruct(dataset):
    tokens = OrderedDict()
    for sentences in dataset:
        s_tokens = sentences.split()
        for t in s_tokens:
            if t in tokens:
                tokens[t] += 1
            else:
                tokens[t] = 1
    return tokens

tokens = BoWStruct(dataset)

def getVector(sentence):
    v = np.zeros(len(tokens) + 1)
    for t in sentence.split():
        if t in tokens:
            v[list(tokens.keys()).index(t)] += 1
        else:
            v[-1] += 1
    return v.tolist()

ex1 = 'i enjoy spending time with my big cat'
vecs = {0:[],1:[]}
data0 = ['i have a big cat .', 'my cat is very big .', 'i enjoy spending time with my cat !']
data1 = ['dogs are quite nice , i enjoy spending time with them !', 'nice dog .']
for i in data0:
    vecs[0].append(getVector(i))
for i in data1:
    vecs[1].append(getVector(i))

nb.summaries = nb.summarize(vecs)
print(nb.calcClassProbabilities(nb.summaries, getVector('i have cat !')))