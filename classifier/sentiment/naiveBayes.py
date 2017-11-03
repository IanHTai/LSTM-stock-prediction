import scipy.stats
import numpy as np
from dataExtract.dataExtractor import *
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple, OrderedDict
from classifier.sentiment.classifiers import Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

class NaiveBayes(Classifier):
    def __init__(self):
        pass
    def summarize(self, separated_dataset):
        summaries = {}
        for classValue, instances in separated_dataset.items():
            npArr = np.array(instances)
            means = npArr.mean(axis=0, dtype=np.float64)
            stdevs = npArr.std(axis=0, dtype=np.float64)
            summaries[classValue] = list(zip(means, stdevs))
        return summaries

    def probability(self, x, mean, stdev):
        return scipy.stats.norm(mean, stdev).pdf(x)

    def calcLogClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 0
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] += np.log(self.probability(x, mean, stdev))
        return probabilities

    def calcScaledClassProbs(self, probabilities):
        sum = 0
        listProb = np.exp(np.negative(list(probabilities.keys())))
        for prob in listProb:
            sum += prob
        out = [0]*len(listProb)
        for i in range(len(listProb)):
            out[i] = listProb[i]/sum
        return out

    def getTrainInputVectors(self):
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\training_set')

        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        self.docs = []
        values = list(conEx.process().values())
        combined = values[0] + values[1]

        for i, text in enumerate(combined):
            words = text.lower().split()
            tags = [i]
            self.docs.append(analyzedDocument(words, tags))

        self.model = Doc2Vec(dm=0, size=100, window=10, min_count=5, workers=4, iter=5)
        self.model.build_vocab(self.docs)
        self.model.train(self.docs, total_examples=2740, epochs=self.model.iter)

        len1 = len(values[0])
        len2 = len(values[1])

        sepVecs = {0:[], 1:[]}
        for i in range(0, len1 + len2):
            if i < len(values[0]):
                sepVecs[0].append(self.model.docvecs[i])
            else:
                sepVecs[1].append(self.model.docvecs[i])
        return sepVecs


    def getTestInputVectors(self):
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\test_set')

        values = list(conEx.process().values())
        sepArrs = []
        for i in values[0]:
            sepArrs.append([self.model.infer_vector(i), 0])

        for i in values[1]:
            sepArrs.append([self.model.infer_vector(i), 1])

        return sepArrs

    def giveProbs(self, summaries, sentence):
        return self.calcLogClassProbabilities(summaries, self.model.infer_vector(sentence))

    def train(self, trainArr):
        self.summaries = self.summarize(trainArr)

    def test(self, testArr):
        return (self.predict(self.summaries, testArr[0])) == testArr[1]

    def testBatch(self, testArrs):
        total = 0.
        correct = 0.
        for i in testArrs:
            total += 1.
            if(self.test(i)):
                correct += 1.
        return correct/total

    def predict(self, summaries, sentence):
        logprobs = self.calcLogClassProbabilities(summaries, sentence)
        return np.argmax(logprobs)


nb = NaiveBayes()
nb.train(nb.getTrainInputVectors())
print('Accuracy: ', nb.testBatch(nb.getTestInputVectors()))

'''

===================================
Test against SKLearn's classifiers
===================================

'''
data = nb.model.docvecs
newOnes = np.zeros(len(data))
temp = nb.getTrainInputVectors()
for i in range(len(temp[0]), len(temp[1])):
    newOnes[i] = 1
testData = list(zip(*(nb.getTestInputVectors())))


gnb = GaussianNB()
gnb.fit(data, newOnes)
print('SKL Gaussian NB score: ', gnb.score(testData[0], testData[1]))

perc = Perceptron(max_iter=1000)
perc.fit(data, newOnes)
print('SKL Perceptron score: ', perc.score(testData[0], testData[1]))




"""
    print(nb.calcScaledClassProbs(nb.calcLogClassProbabilities(nb.summaries, nb.model.docvecs[50])))
    print(nb.calcScaledClassProbs(nb.calcLogClassProbabilities(nb.summaries, nb.model.docvecs[2500])))
    #print(nb.model.docvecs[50], nb.model.docvecs[2500])

    print('\n\n')
    print(gnb.predict_proba([data[50]]), nb.docs[50])
    print(gnb.predict_proba([data[51]]), nb.docs[51])
    print(gnb.predict_proba([data[52]]), nb.docs[52])
    print(gnb.predict_proba([data[53]]), nb.docs[53])
    print(gnb.predict_proba([data[54]]), nb.docs[54])
    print(gnb.predict_proba([data[55]]), nb.docs[55])
    
    
    
    print(gnb.predict_proba([data[2500]]), nb.docs[2501])
    print(gnb.predict_proba([data[2501]]), nb.docs[2502])
    print(gnb.predict_proba([data[2502]]), nb.docs[2503])
    print(gnb.predict_proba([data[2503]]), nb.docs[2504])
    print(gnb.predict_proba([data[2504]]), nb.docs[2505])
    print(gnb.predict_proba([data[2505]]), nb.docs[2506])
    print(gnb.predict_proba([data[2506]]), nb.docs[2507])
    print('News Twitter: ', gnb.predict_proba([nb.model.infer_vector('The European Parliament has warned that " hostile propaganda " by Russia against the EU is growing'.lower())]))
"""

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