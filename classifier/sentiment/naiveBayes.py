import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import scipy.stats
import numpy as np
from dataExtract.dataExtractor import *
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple, OrderedDict
from classifier.sentiment.classifiers import Classifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
import cProfile
import time
from copy import deepcopy
import nltk
import semiSupervised
from sklearn.svm import SVC

class GaussianNaiveBayes(Classifier):
    counts = {}
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
            self.counts[classValue] = len(separated[classValue])
            summaries[classValue] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*separated[classValue])]

        return summaries

    def probability(self, x, mean, stdev):
        return scipy.stats.norm(mean, stdev).pdf(x)

    def calcLogClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = np.log(self.counts[classValue]/float(np.sum(list(self.counts.values()))))
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
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\processed\\convote\\train')

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
        conEx = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\processed\\convote\\test')

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
        return max(logprobs, key=logprobs.get)

class NaiveBayes(GaussianNaiveBayes):
    def predict(self, summaries, sentence):
        logprobs = self.calcLogClassProbabilities(summaries, sentence)
        return max(logprobs, key=logprobs.get)

    def train(self, trainArr):
        self.logsummaries = self.summarize(trainArr)

    def test(self, testArr):
        return str((self.predict(self.logsummaries, testArr[0]))) == testArr[1]

    def testBatch(self, testArrs):
        total = 0.
        correct = 0.
        for i in testArrs:
            total += 1.

            if(self.test(i)):
                correct += 1.

        return correct/total

    def summarize(self, dataset):
        separated = {}

        for input in dataset:
            if input[1] in separated:
                separated[input[1]].append(list(input[0]))
            else:
                separated[input[1]] = [list(input[0])]
        condProbs = {}
        for classValue in separated.keys():
            self.counts[classValue] = len(separated[classValue])
            totalCountVector = np.sum(separated[classValue], axis=0)
            addOne = np.full((totalCountVector.size), 1)
            np.add(totalCountVector, addOne, totalCountVector)
            classCondProbs = []
            for feature in totalCountVector:
                classCondProbs.append(feature/float(np.sum(totalCountVector) - feature))
            condProbs[classValue] = classCondProbs
        self.summaries = condProbs
        self.logsummaries = {}
        for key in condProbs.keys():
            self.logsummaries[key] = np.log(condProbs[key])
        return self.logsummaries

    def calcLogClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue in summaries:
            prior = np.log(self.counts[classValue]/float(np.sum(list(self.counts.values()))))
            probabilities[classValue] = prior + np.dot(inputVector,summaries[classValue])
        return probabilities
def BoWStruct(dataset):
    tokens = OrderedDict()
    for sentences in dataset:
        s_tokens = sentences.split()
        for t in s_tokens:
            if not t in tokens:
                tokens[t] = 0
    return tokens

def getVector(sentence, struct):
    v = np.zeros(len(struct))
    keysList = list(struct.keys())
    for word in sentence.split():
        if word in struct:
            v[keysList.index(word)] += 1
    return v

def doc2Vec(data, train=True, d2v_model=None):
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    docs = []
    if (train):

        for i, text in enumerate(data):
            words = text.lower().split()
            tags = [i]
            docs.append(analyzedDocument(words, tags))

        model = Doc2Vec(dm=0, size=100, window=10, min_count=5, workers=4, iter=10)
        model.build_vocab(docs)
        model.train(docs, total_examples=len(data), epochs=model.iter)
        newValues = []
        for input in data:
            tokens = input.lower().split()
            fv = np.array(model.infer_vector(tokens), dtype='float64')
            newValues.append(fv)
        return np.array(newValues), model
    else:
        assert(not (d2v_model == None))
        for i, text in enumerate(data):
            words = text.lower().split()
            tags = [i]
            docs.append(analyzedDocument(words, tags))
        newValues = []
        for input in data:
            tokens = input.lower().split()
            fv = np.array(d2v_model.infer_vector(tokens), dtype='float64')
            newValues.append(fv)
        return np.array(newValues), d2v_model


def getConVoteData(train=True):
    tokens = {}
    if(train):
        CE = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\processed\\convote\\train')
        values = CE.process()
        tokens = BoWStruct(values[:, 0])
    else:
        CE = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\processed\\convote\\test')
        values = CE.process()
    return values,tokens

if __name__ == '__main__':

    ss = semiSupervised.SemiSupervised()
    data, labels = ss.getTwitterRaw('../../resources/hydrated_tweets/relevant_tweets.txt')

    # SHUFFLING DATA
    data, labels = shuffle(data, labels)

    trainPercent = 0.8
    trainData = data[0:int(trainPercent*len(data))]
    trainLabels = labels[0:int(trainPercent*len(labels))]
    assert(len(trainData) == len(trainLabels))

    testData = data[int(trainPercent*len(data)):]
    testLabels = labels[int(trainPercent*len(labels)):]
    assert(len(trainData) == len(trainLabels))

    structure = BoWStruct(trainData)
    trainFV = []
    for sentence, value in zip(trainData, trainLabels):
        trainFV.append(getVector(sentence, structure))
    trainFV = np.array(trainFV)

    testFV = []
    for sentence, value in zip(testData, testLabels):
        testFV.append(getVector(sentence, structure))
    testFV = np.array(testFV)


    # Cross Validation for hyperparameter tuning
    paramDict = {
    }
    c_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_values = [0.1, 0.5, 1, 5, 10]
    kernel = 'rbf'

    for C in c_values:
        for gamma in gamma_values:
            BoWSVM = SVC(C=C, kernel=kernel, gamma=gamma)
            meanScore = np.mean(cross_val_score(BoWSVM, trainFV, trainLabels, cv=10, n_jobs=-1))
            print(C, gamma, meanScore)
            paramDict[meanScore] = [C,gamma]

    sortedParams = sorted(list(paramDict.keys()), reverse=True)
    print(sortedParams)
    params = paramDict[sortedParams[0]]

    BoWSVM = SVC(C=params[0], kernel='rbf', gamma=params[1])
    BoWSVM.fit(trainFV, trainLabels)

    BoWNB = MultinomialNB()
    BoWNB.fit(trainFV, trainLabels)

    print("SVM Score", BoWSVM.score(testFV, testLabels))
    print("NB Score", BoWNB.score(testFV, testLabels))

    '''
    ===================================
    Bag of Words
    ===================================
    '''
    # print('Getting BoW Train Data')
    # trainValues, structure = getConVoteData(train=True)
    # trainFV = []
    # for [sentence, value] in trainValues:
    #     trainFV.append([getVector(sentence, structure), value])
    # trainFV = np.array(trainFV)
    # print('Training BoW')
    # BoWNB = NaiveBayes()
    # BoWNB.train(trainFV)
    # testValues, emptyStruct = getConVoteData(train=False)
    # print('Getting BoW Test Data')
    # testFV = []
    # for [sentence, value] in testValues:
    #     testFV.append([getVector(sentence, structure), value])
    # testFV = np.array(testFV)
    # print('Testing BoW')
    # print('BoW Accuracy: ', BoWNB.testBatch(testFV))
    #
    #
    #
    #
    # '''
    # ===================================
    # Doc2Vec
    # ===================================
    # '''
    # print('Getting Doc2Vec Train Data')
    # nb = GaussianNaiveBayes()
    # print('Training Doc2Vec NB')
    # nb.train(nb.getTrainInputVectors())
    # print('Getting Doc2Vec Test Data')
    # test = nb.getTestInputVectors()
    # currTime = time.time()
    # print('Testing Doc2Vec')
    # print('Accuracy: ', nb.testBatch(test))
    # '''
    #
    # ===================================
    # Test against SKLearn's classifiers
    # ===================================
    #
    # '''
    # trainSet = nb.getTrainInputVectors()
    #
    # gnb = GaussianNB()
    # gnb.fit(list(trainSet[:,0]), trainSet[:,1])
    # print('SKL Gaussian NB score: ', gnb.score(list(test[:,0]), test[:,1]))
    #
    # perc = Perceptron(max_iter=1000)
    # perc.fit(list(trainSet[:,0]), trainSet[:,1])
    # print('SKL Perceptron score: ', perc.score(list(test[:,0]), test[:,1]))
