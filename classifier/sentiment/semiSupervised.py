from sklearn.semi_supervised import LabelSpreading
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import datasets
import numpy as np
from sklearn.model_selection import GridSearchCV
import random
from featureGen.sentimentFeatures import SentimentFeatures

class SemiSupervised:
    def getD2VTrainVectors(self, combinedPath):
        self.model = Doc2Vec(dm=0, size=200, window=10, workers=4, iter=10)
        sentences = []
        with open(combinedPath, 'r') as combined:
            for line in combined:
                splitLine = line.rstrip('\n').split('\t')
                if len(splitLine) == 4:
                    sentences.append(TaggedDocument(words=splitLine[2].split(), tags=[splitLine[0]+' '+splitLine[1]]))
        print('finished getting sentences')
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=1386047, epochs=self.model.iter)
        print('trained doc2vec')
        dataset = []
        labels = []
        with open(combinedPath, 'r') as combined:
            counter = 0
            for line in combined:
                splitLine = line.rstrip('\n').split('\t')
                if len(splitLine) == 4:
                    if(splitLine[3] == '-1'):
                        dataset.append(self.model.infer_vector(splitLine[2]))
                        labels.append(-1)

                    elif(splitLine[3] == 'neg'):
                        dataset.append(self.model.infer_vector(splitLine[2]))
                        labels.append(0)
                    elif (splitLine[3] == 'neutral'):
                        dataset.append(self.model.infer_vector(splitLine[2]))
                        labels.append(5)
                    elif (splitLine[3] == 'pos'):
                        dataset.append(self.model.infer_vector(splitLine[2]))
                        labels.append(10)
                    counter += 1

        print('got vectors from doc2vec')
        #self.model.save('/tmp/twitter-model')
        #self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        return np.array(dataset),np.array(labels)
    def getD2VTestVectors(self, testPath):
        sentences = []
        dataset = []
        with open(testPath, 'r') as test:
            for line in test:
                splitLine = line.rstrip('\n').split('\t')
                if len(splitLine) == 4:
                    if (splitLine[3] == '-1'):
                        dataset.append(np.array([self.model.infer_vector(splitLine[2]), -1]))
                    elif (splitLine[3] == 'neg'):
                        dataset.append(np.array([self.model.infer_vector(splitLine[2]), 0]))
                    elif (splitLine[3] == 'neutral'):
                        dataset.append(np.array([self.model.infer_vector(splitLine[2]), 5]))
                    elif (splitLine[3] == 'pos'):
                        dataset.append(np.array([self.model.infer_vector(splitLine[2]), 10]))
        print('got test vectors')
        return np.array(dataset)

    def getTwitterRaw(self, combinedPath):
        sentences = []
        labels = []
        with open(combinedPath, 'r') as combined:
            for line in combined:
                splitLine = line.rstrip('\n').split('\t')
                if len(splitLine) == 4:
                    if (splitLine[3] == '-1'):
                        sentences.append(splitLine[2])
                        labels.append(-1)
                    elif (splitLine[3] == 'neg'):
                        sentences.append(splitLine[2])
                        labels.append(0)
                    elif (splitLine[3] == 'neutral'):
                        sentences.append(splitLine[2])
                        labels.append(5)
                    elif (splitLine[3] == 'pos'):
                        sentences.append(splitLine[2])
                        labels.append(10)
        return np.array(sentences), np.array(labels)


def getXORDataset(numUnlabelled, numLabelled):
    unlab = []
    for i in range(numUnlabelled):
        bit1 = random.randint(0,1)
        bit2 = random.randint(0,1)
        unlab.append([[bit1, bit2], -1])
    lab = []
    for i in range(numLabelled):
        bit1 = random.randint(0, 1)
        bit2 = random.randint(0, 1)
        xor = bit1 ^ bit2
        lab.append([[bit1, bit2], xor])

    return unlab, lab

def getIrisDataset(numUnlabelled, numLabelled):
    iris = datasets.load_iris()

    rng = np.random.RandomState(random.randint(0, 100))
    unlabelled_indices = rng.rand(len(iris.target)) < float(numLabelled)/(float(numUnlabelled + numLabelled))
    unlabels = np.copy(iris.target)
    unlabels[unlabelled_indices] = -1

    lab = []
    for i in range(len(iris.target)):
        if(random.random() < float(numLabelled)/(float(numUnlabelled + numLabelled))):
            lab.append([list(iris.data[i]), iris.target[i]])
    #unlab = list(data[unlabelled_indices])
    #for i in range(len(unlab)):
    #    unlab[i] = [list(unlab[i]), -1]
    #lab = []
    #labelledData = data[np.logical_not(unlabelled_indices)]
    #labelledLabels = labels[np.logical_not(unlabelled_indices)]

    #for i in range(len(labelledData)):
    #    lab.append([list(labelledData[i]), labelledLabels[i]])
    #lab = np.concatenate((data[np.logical_not(unlabelled_indices)])[...,np.newaxis], (labels[np.logical_not(unlabelled_indices)])[...,np.newaxis])
    return iris.data, unlabels, lab



if __name__ == '__main__':

    #unlab, lab = getXORDataset(5000, 300)
    #data, unlab, lab = getIrisDataset(5000, 300)
    #trainData = np.array(unlab + lab[:int(len(lab)*2/3)])
    #testData = np.array(lab[int(len(lab)*2/3):])
    #testData = np.array(lab)
    #print(testData)

    ss = SemiSupervised()
    #data,unlab = ss.getD2VTrainVectors('../../resources/hydrated_tweets/small_data/Combined_subsampled_Dev.txt')

    sf = SentimentFeatures()
    rawdata,unlab = ss.getTwitterRaw('../../resources/hydrated_tweets/small_data/Combined_subsampled_Dev.txt')
    print('Number of unique words: ', len(sf.POSStruct(rawdata)))
    data = []
    for sentence in rawdata:
        data.append(sf.genAdjVec(sentence))
    ### print('got Vectors')
    ### model = LabelSpreading(kernel='rbf')
    ### params = {'gamma':[0.01,0.1,1.0,10.0,100.0], 'max_iter':[10,100,1000], 'alpha':[0.2,0.4,0.6,0.8]}
    #gridsearch = GridSearchCV(model, params, n_jobs=3 )

    #gridsearch.fit(list(data[:,0]), data[:,1].astype('int'))
    #print(gridsearch.cv_results_)
    #print('model fitted')
    #testData = ss.getD2VTestVectors('../../resources/hydrated_tweets/small_data/Test_Dev.txt')
    testDataRaw, testLabels = ss.getTwitterRaw('../../resources/hydrated_tweets/small_data/Test_Dev.txt')
    testData = []
    for sentence in testDataRaw:
        testData.append(sf.genAdjVec(sentence))
    ###scoreDict = {}
###
    ###for gamma in params['gamma']:
    ###    model.gamma = gamma
    ###    for max_iter in params['max_iter']:
    ###        model.max_iter = max_iter
    ###        for alpha in params['alpha']:
    ###            model.alpha = alpha
    ###            model.fit(list(trainData[:,0]), trainData[:,1].astype('int'))
    ###            score = model.score(list(testData[:, 0]), testData[:,1].astype('int'))
    ###            print(score, ' gamma = ', gamma, ' max_iter = ', max_iter, ' alpha = ', alpha)
    ###            if(score in scoreDict):
    ###                scoreDict[score].append('gamma = ' + str(gamma) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha))
    ###            else:
    ###                scoreDict[score] = ['gamma = ' + str(gamma) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha)]
###
    ###knnModel = LabelSpreading(kernel='knn')
    ###knnParams = {'n_neighbors':[1, 4, 9, 16], 'max_iter':[10,100,1000], 'alpha':[0.2,0.4,0.6,0.8]}
###
    ###for n_neighbors in knnParams['n_neighbors']:
    ###    model.n_neighbors = n_neighbors
    ###    for max_iter in knnParams['max_iter']:
    ###        model.max_iter = max_iter
    ###        for alpha in knnParams['alpha']:
    ###            model.alpha = alpha
    ###            model.fit(list(trainData[:,0]), trainData[:,1].astype('int'))
    ###            score = model.score(list(testData[:, 0]), testData[:,1].astype('int'))
    ###            print(score, ' n_neighbors = ', n_neighbors, ' max_iter = ', max_iter, ' alpha = ', alpha)
    ###            if(score in scoreDict):
    ###                scoreDict[score].append('n_neighbors = ' + str(n_neighbors) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha))
    ###            else:
    ###                scoreDict[score] = ['n_neighbors = ' + str(n_neighbors) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha)]
###
###
    ###fastest = sorted(list(scoreDict.keys()))[-1]
    ###print(fastest, scoreDict[fastest])

    #print(gridsearch.score(list(testData[:, 0]), testData[:,1].astype('int')))
    #print(gridsearch.best_params_)

    print('got Vectors')
    model = LabelSpreading(kernel='rbf')
    params = {'gamma': [0.1, 1.0, 10.0, 30.0, 50.0, 80.0, 100.0, 300.0], 'max_iter': [10, 100, 1000], 'alpha': [0.2, 0.4, 0.6, 0.8]}

    scoreDict = {}

    for max_iter in params['max_iter']:
        model.max_iter = max_iter
        for alpha in params['alpha']:
            model.alpha = alpha
            for gamma in params['gamma']:
                model.gamma = gamma
                model.fit(list(data), list(unlab))
                score = model.score(list(testData), list(testLabels))
                print(score, ' gamma = ', gamma, ' max_iter = ', max_iter, ' alpha = ', alpha)
                if (score in scoreDict):
                    scoreDict[score].append(
                        'gamma = ' + str(gamma) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha))
                else:
                    scoreDict[score] = [
                        'gamma = ' + str(gamma) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha)]

    knnModel = LabelSpreading(kernel='knn')
    knnParams = {'n_neighbors': [1, 4, 9, 16], 'max_iter': [10, 100, 1000], 'alpha': [0.2, 0.4, 0.6, 0.8]}
    for max_iter in knnParams['max_iter']:
        model.max_iter = max_iter
        for alpha in knnParams['alpha']:
            model.alpha = alpha
            for n_neighbors in knnParams['n_neighbors']:
                model.n_neighbors = n_neighbors
                model.fit(list(data), list(unlab))
                score = model.score(list(testData), list(testLabels))
                print(score, ' n_neighbors = ', n_neighbors, ' max_iter = ', max_iter, ' alpha = ', alpha)
                if (score in scoreDict):
                    scoreDict[score].append(
                        'n_neighbors = ' + str(n_neighbors) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha))
                else:
                    scoreDict[score] = [
                        'n_neighbors = ' + str(n_neighbors) + ' max_iter = ' + str(max_iter) + ' alpha = ' + str(alpha)]

    best = sorted(list(scoreDict.keys()))[-1]
    print('Best accuracy: ', best, scoreDict[best])