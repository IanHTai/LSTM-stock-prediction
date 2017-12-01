from classifier.sentiment.naiveBayes import BoWStruct, getVector
import numpy as np
from collections import OrderedDict
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn



#nltk.download()
class SentimentFeatures:
    # FeatureVector Length
    FVLength = 6
    #def __init__(self):

    def buildStruct(self, dataset):
        self.struct = BoWStruct(dataset)
        return self.struct

    def genVec(self, sentence):
        return getVector(sentence, self.struct)

    def genPresenceVec(self, sentence):
       v = np.zeros(len(self.struct))
       keysList = list(self.struct.keys())
       for word in sentence.split():
           if word in self.struct:
               v[keysList.index(word)] = 1
       return v

    POSDict = {
        'JJ':'a',
        'JJR':'a',
        'JJS':'a',
        'NN':'n',
        'NNS':'n',
        'NNP':'n',
        'NNPS':'n',
        'RB':'av',
        'RBR':'av',
        'RBS':'av'
    }

    def genAdjVec(self, sentence):
        v = np.zeros(len(self.adjStruct))
        s_tokens = nltk.word_tokenize(sentence)
        pos_tagged = nltk.pos_tag(s_tokens)
        adjCount = 0
        adjTotalScore = np.zeros(3)

        advCount = 0
        advTotalScore = np.zeros(3)
        for (word, pos) in pos_tagged:
            if pos in self.POSDict:
                if self.POSDict[pos] == 'a':
                    if word in self.adjStruct:
                        v[self.adjKeyList.index(word)] = v[self.adjKeyList.index(word)] + 1.

                    np.add(adjTotalScore, self.wordScore(word, 'a'), adjTotalScore)
                    adjCount += 1
                elif self.POSDict[pos] == 'av':
                    np.add(advTotalScore, self.wordScore(word, 'a'), advTotalScore)
                    advCount += 1
        adjDiv = np.zeros(3)
        advDiv = np.zeros(3)
        if(adjCount != 0):
            np.true_divide(adjTotalScore, adjCount, adjDiv)
        if (advCount != 0):
            np.true_divide(advTotalScore, advCount, advDiv)
        nums = np.append(adjDiv, advDiv)

        v = np.concatenate((v, nums))
        return v
    def POSStruct(self, dataset):
        tokens = OrderedDict()
        for sentence in dataset:
            s_tokens = nltk.word_tokenize(sentence)
            pos_tagged = nltk.pos_tag(s_tokens)
            for (word, pos) in pos_tagged:
                if pos == 'JJ':
                    if not word in tokens:
                        tokens[word] = 1
                    else:
                        tokens[word] += 1
        self.adjStruct = tokens
        self.adjKeyList = list(self.adjStruct.keys())
        return tokens

    def wordScore(self, adj, pos):
        if(pos == 'a'):
            synset = wn.synsets(adj, pos=wn.ADJ)
            if(len(synset) >= 1):
                breakdown = swn.senti_synset(synset[0].name())
                return np.array([breakdown.pos_score(), breakdown.neg_score(), breakdown.obj_score()])
        elif (pos == 'av'):
            synset = wn.synsets(adj, pos=wn.ADV)
            if (len(synset) >= 1):
                breakdown = swn.senti_synset(synset[0].name())
                return np.array([breakdown.pos_score(), breakdown.neg_score(), breakdown.obj_score()])
        return np.array([0,0,0])