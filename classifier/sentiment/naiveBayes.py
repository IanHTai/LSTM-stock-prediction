import scipy.stats
import numpy as np
from dataExtract.dataExtractor import *
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple, OrderedDict
from classifier.sentiment.classifiers import Classifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import Perceptron
import cProfile
import time

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
        return np.argmax(list(logprobs.values()))

class NaiveBayes(GaussianNaiveBayes):
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
        return self.summaries

    def calcLogClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue in summaries:
            probabilities[classValue] = np.log(self.counts[classValue]/float(np.sum(list(self.counts.values()))))
            position = 0
            for feature in (inputVector):
                for i in range(feature):
                    probabilities[classValue] += np.log(summaries[classValue][position])
                position += 1
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
    v = struct
    for word in sentence.split():
        if word in v:
            v[word] += 1
    return list(v.values())

def getConVoteData(train=True):
    tokens = {}
    if(train):
        CE = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\training_set')
        values = CE.process()
        tokens = BoWStruct(values[:, 0])
    else:
        CE = ConVoteExtractor(os.getcwd() + '\\..\\..\\resources\\raw\\convote_v1.1\\data_stage_three\\test_set')
        values = CE.process()
    return values,tokens

'''
===================================
Bag of Words
===================================
'''
print('Getting BoW Train Data')
trainValues, structure = getConVoteData(train=True)
trainFV = []
for [sentence, value] in trainValues:
    trainFV.append([getVector(sentence, structure), value])
trainFV = np.array(trainFV)
print('Training BoW')
BoWNB = NaiveBayes()
BoWNB.train(trainFV)
testValues, emptyStruct = getConVoteData(train=False)
print('Getting BoW Test Data')
testFV = []
for [sentence, value] in testValues:
    testFV.append([getVector(sentence, structure), value])
testFV = np.array(testFV)
print('Testing BoW')
print('BoW Accuracy: ', BoWNB.testBatch(testFV))




'''
===================================
Doc2Vec
===================================
'''
nb = GaussianNaiveBayes()
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