import random
import numpy as np

class randomAgent:
    def __init__(self, stdev, data):
        self.stdev = stdev
        self.data = data
    def predictions(self):
        preds = []
        for previous in self.data:
            preds.append(previous + random.gauss(0,self.stdev))
        return np.array(preds)
class martingale(randomAgent):
    def __init__(self, data):
        self.data = data

    def predictions(self):
        return self.data

def calcPerformance(lstm, loss, acc):
    scaledFeats = lstm.test_X[:, :, 0]
    targets = lstm.test_Y[:, 0]

    randomPerf = []

    for i in range(0, 5000):
        stdev = random.gauss(1, 0.3)
        r = randomAgent(stdev, scaledFeats)
        preds = r.predictions()

        MSE = np.square(preds - targets).mean()

        last = lstm.test_X[0, 0, 0]
        numCorrect = 0
        total = len(targets)
        assert(len(preds) == len(targets))
        for i in range(0, len(targets)):
            if (np.sign(preds[i] - last) == np.sign(targets[i] - last)):
                numCorrect += 1
            last = targets[i]
        randomPerf.append([MSE, float(numCorrect)/float(total)])

    r = martingale(scaledFeats)
    preds = r.predictions()
    MSE = np.square(preds - targets).mean()
    assert (len(preds) == len(targets))
    martingalePerf = MSE

    numBetterLoss = 0
    numBetterAcc = 0
    totalNum = len(randomPerf)
    for i in range(0, len(randomPerf)):
        if loss < randomPerf[i][0]:
            numBetterLoss += 1
        if acc > randomPerf[i][1]:
            numBetterAcc += 1

    return numBetterLoss, numBetterAcc, totalNum, martingalePerf