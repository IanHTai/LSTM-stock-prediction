import scipy.stats
import numpy as np
from dataExtract.dataExtractor import *
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple, OrderedDict
from classifier.sentiment.classifiers import Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
import cProfile
import time

class NaiveBayes(Classifier):
    def __init__(self):
        pass
    def summarize(self, dataset):
        summaries = {}

        separated = {}

        for input in dataset:
            if input[1] in separated:
                separated[input[1]].append(list(input[0]))
            else:
                separated[input[1]] = [list(input[0])]

        for classValue in separated.keys():
            summaries[classValue] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*separated[classValue])]

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
        values = conEx.process()


        for i, text in enumerate(values[:,0]):
            words = text.lower().split()
            tags = [i]
            self.docs.append(analyzedDocument(words, tags))

        self.model = Doc2Vec(dm=0, size=100, window=10, min_count=5, workers=4, iter=10)
        self.model.build_vocab(self.docs)
        self.model.train(self.docs, total_examples=2740, epochs=self.model.iter)
        newValues = []
        for input in values:
            tokens = input[0].lower().split()
            fv = np.array(self.model.infer_vector(tokens), dtype='float64')
            newValues.append(np.array([fv, input[1]]))

        return np.array(newValues)

    def getTestInputVectors(self):
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\test_set')

        values = conEx.process()

        newValues = []
        for [sentence, value] in values:
            newValues.append([self.model.infer_vector(sentence.lower().split()), value])

        return np.array(newValues)

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
        currentHighestKey = 0
        currentHighestProb = float("-inf")
        for key in logprobs.keys():
            if logprobs[key] > currentHighestProb:
                currentHighestProb = logprobs[key]
                currentHighestKey = key
        return currentHighestKey


def BoWStruct(dataset):
    tokens = OrderedDict()
    for sentences in dataset:
        s_tokens = sentences.split()
        for t in s_tokens:
            if not t in tokens:
                tokens[t] = 0
    return tokens
CE = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\training_set')
values = CE.process()
tokens = BoWStruct(values[:,0])
print(len(tokens))



def getVector(sentence):
    v = np.zeros(len(tokens) + 1)
    for t in sentence.split():
        if t in tokens:
            v[list(tokens.keys()).index(t)] += 1
        else:
            v[-1] += 1
    return v.tolist()


nb = NaiveBayes()
nb.train(nb.getTrainInputVectors())
test = nb.getTestInputVectors()
currTime = time.time()
print('Accuracy: ', nb.testBatch(test))
print('Test Time: ', time.time() - currTime)
'''

===================================
Test against SKLearn's classifiers
===================================

'''
trainSet = nb.getTrainInputVectors()

gnb = GaussianNB()
gnb.fit(list(trainSet[:,0]), trainSet[:,1])
print('SKL Gaussian NB score: ', gnb.score(list(test[:,0]), test[:,1]))

perc = Perceptron(max_iter=1000)
perc.fit(list(trainSet[:,0]), trainSet[:,1])
print('SKL Perceptron score: ', perc.score(list(test[:,0]), test[:,1]))


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