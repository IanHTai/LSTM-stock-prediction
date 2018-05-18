from classifier.sentiment import naiveBayes
from classifier.main import lstm, randomAgents
import tensorflow as tf
import numpy as np
from dataExtract.dataExtractor import FinExtractor
import os
import datetime
import random

if __name__ == '__main__':
    print("Time at start of program: ", datetime.datetime.now())

    # Save or not
    trained = False
    bestParams = []
    useSenti = False


    STOCK = 'AAPL'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    n = lstm.LSTM_System(handleData=False)
    n.config.stock = STOCK

    finExtractor = FinExtractor()
    tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
    n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
    n.df = n.convertDates(n.df)
    n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

    # IMPORTANT: REMOVE EVERY 2ND ROW
    n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

    # REMOVE EVERY 2ND ROW AGAIN
    n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

    splitRatio = n.config.test_train_split
    splitDate = n.df[int(splitRatio * len(n.df)), 0]

    #TEMP
    print(splitDate)

    sentiDates, sentiData, sentiFeats, sentiClass = naiveBayes.getSentiClassifier(lastTrainDate=splitDate, newTraining=False)


    """
    Separating twitter dataset to individual stocks
    """

    #dict of dates and classified labels of relevant tweets to each company
    separated = {
        'AAPL': [],
        'AMZN': [],
        'GOOG': [],
        'GOOGL': [],
        'MSFT': []
    }
    AAPLKeyWords = ['aapl', 'apple']
    AMZNKeyWords = ['amzn', 'amazon']
    GOOGKeyWords = ['goog', 'alphabet', 'google']
    GOOGLKeyWords = ['googl', 'alphabet', 'google']
    MSFTKeyWords = ['msft', 'microsoft']

    for i in range(0, len(sentiData)):
        if any(substring in sentiData[i] for substring in AAPLKeyWords):
            feature = sentiFeats[i].reshape(1, -1)
            toAdd = float(sentiClass.predict(feature))
            separated['AAPL'].append([sentiDates[i], toAdd])
        if any(substring in sentiData[i] for substring in AMZNKeyWords):
            feature = sentiFeats[i].reshape(1, -1)
            toAdd = float(sentiClass.predict(feature))
            separated['AMZN'].append([sentiDates[i], toAdd])
        if any(substring in sentiData[i] for substring in GOOGKeyWords):
            feature = sentiFeats[i].reshape(1, -1)
            toAdd = float(sentiClass.predict(feature))
            separated['GOOG'].append([sentiDates[i], toAdd])
        if any(substring in sentiData[i] for substring in GOOGLKeyWords):
            feature = sentiFeats[i].reshape(1, -1)
            toAdd = float(sentiClass.predict(feature))
            separated['GOOGL'].append([sentiDates[i], toAdd])
        if any(substring in sentiData[i] for substring in MSFTKeyWords):
            feature = sentiFeats[i].reshape(1, -1)
            toAdd = float(sentiClass.predict(feature))
            separated['MSFT'].append([sentiDates[i], toAdd])

    if not trained:
        """
        Hyper parameter tuning

        Due to the time it takes for this process, we will only be tuning for 1 stock and then using
        those hyperparameters for all stocks
        """

        #TEMPORARY
        if useSenti:

            errorDict = {}
            # Random search
            for i in range(0, 10):
                tf.reset_default_graph()
                n = lstm.LSTM_System(handleData=False)
                n.config.stock = STOCK

                finExtractor = FinExtractor()
                tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
                n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
                n.df = n.convertDates(n.df)
                n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

                # IMPORTANT: REMOVE EVERY 2ND ROW
                n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

                # REMOVE EVERY 2ND ROW AGAIN
                n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

                n.config.num_layers = random.randint(2,5)  # np.random.choice(3, p=[0.5, 0.25, 0.25]) + 2
                n.config.batch_size = 1  # np.random.randint(1,300/n.config.num_layers)
                # n.config.num_steps = np.random.randint(1,n.config.batch_size+1)
                n.config.keep_prob = np.random.uniform(0.3, 0.8)
                n.config.lstm_size = np.random.randint(8, 512)
                n.config.init_learning_rate = np.random.uniform(0.0005, 0.001)
                n.config.max_epoch = int(float(1000) / n.config.num_layers)
                n.config.early_stopping_continue = int(float(40) / n.config.num_layers)
                n.config.williams_period = np.random.randint(10, 20)
                n.config.avgAlpha = np.random.uniform(0.1, 0.5)
                n.config.learning_rate_decay = np.random.uniform(0.96, 0.99)
                n.config.train_validation_split = 0.8

                # Currently just testing microsoft
                # n.config.stock = n._UsedStocks[np.random.randint(0,len(n._UsedStocks))]
                print("\n\n")
                print("New hyperparameter test ", i + 1)
                print("Batch Size: ", n.config.batch_size)
                print("Number of Steps: ", n.config.num_steps)
                print("Keep Prob (1-Dropout): ", n.config.keep_prob)
                print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
                print("Number of Hidden Layers: ", n.config.num_layers)
                print("Initial Learning Rate: ", n.config.init_learning_rate)
                print("Williams Period: ", n.config.williams_period)
                print("Average Alpha: ", n.config.avgAlpha)
                print("Learning Rate Decay: ", n.config.learning_rate_decay)
                print("Max epochs: ", n.config.max_epoch)
                print("Early Stopping Epochs Leeway: ", n.config.early_stopping_continue)
                print("\n")

                if useSenti:
                    n.combineInputs(separated[STOCK])
                else:
                    train_size = int(len(n.df) * (n.config.test_train_split))
                    n.df = n.df[0:train_size]
                    n.df = np.delete(n.df, 0, axis=1)
                    n.config.input_size = n.config.input_size - 1
                    stats_feat = n.statsFeatures(n.df, n.config)
                    n.full = n.normPrice(stats_feat, n.config)
                lstm_graph = n.rnn(config=n.config)
                n.train_lstm(lstm_graph, config=n.config)
                errorDict[n.performance] = [n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size,
                                            n.config.num_layers, n.config.init_learning_rate, n.config.williams_period,
                                            n.config.avgAlpha,n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue]

                print("\nDONE\n", n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size,
                      n.config.num_layers,
                      n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha)
            keyList = (sorted(list(errorDict.keys()), reverse=False))
            for i in range(len(keyList)):
                print(errorDict[keyList[i]], "Loss: ", keyList[i])

            bestParams = errorDict[keyList[0]]
        else:
            bestParams = [1, 1, 0.6771585819649831, 469, 5, 0.0008245293691952336, 18, 0.17348624553606168, 0.9760143793838009, 200, 8]

        with open(str(datetime.datetime.now()).replace(" ", "_").replace(":","-") + '_results.txt', 'w') as resultFile:
            if useSenti:
                resultFile.write("HYPERPARAMETER TUNING RESULTS ===============\n")
                for i in range(len(keyList)):
                    resultFile.write(str(errorDict[keyList[i]]) + "Loss: " + str(keyList[i]) + "\n")

                resultFile.write("\n\n\n\n\n\n")
            else:
                resultFile.write("NOT USING SENTIMENT ========================")
            """
            AAPL
            """

            print("\nTRAINING AAPL\n")
            tf.reset_default_graph()
            if useSenti:
                n = lstm.LSTM_System(handleData=False, saveModel=True)
            else:
                n = lstm.LSTM_System(handleData=False, saveModel=False)
            n.config.stock = 'AAPL'

            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue = bestParams

            print("Batch Size: ", n.config.batch_size)
            print("Number of Steps: ", n.config.num_steps)
            print("Keep Prob (1-Dropout): ", n.config.keep_prob)
            print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
            print("Number of Hidden Layers: ", n.config.num_layers)
            print("Initial Learning Rate: ", n.config.init_learning_rate)
            print("Williams Period: ", n.config.williams_period)
            print("Average Alpha: ", n.config.avgAlpha)
            print("Learning Rate Decay: ", n.config.learning_rate_decay)
            print("Max epochs: ", n.config.max_epoch)
            print("\n")


            if useSenti:
                n.combineInputs(separated[STOCK])
            else:
                n.df = np.delete(n.df, 0, axis=1)
                n.config.input_size = n.config.input_size - 1
                stats_feat = n.statsFeatures(n.df, n.config)
                n.full = n.normPrice(stats_feat, n.config)
            lstm_graph = n.rnn(config=n.config)
            n.train_lstm(lstm_graph, config=n.config)
            AAPLLoss = n.performance
            AAPLAcc = n.acc_performance
            print("\nAAPL ========================================")
            resultFile.write("\nAAPL ========================================\n")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            resultFile.write("Loss: " + str(n.performance) + " Directional Accuracy: " + str(n.acc_performance) + "\n")
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            resultFile.write("Performance vs Random Agents:\n")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1], "/", randomPerf[2])
            resultFile.write("Loss better than: " + str(randomPerf[0]) + "/" +
                             str(randomPerf[2]) + " Accuracy better than: " + str(randomPerf[1]) + "/" + str(randomPerf[2]) + "\n")

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            resultFile.write("Mean and Stdev of Random Loss: " + str(randomPerf[4]) + " " + str(randomPerf[5]) + "\n")
            resultFile.write("Mean and Stdev of Random Accuracy: " + str(randomPerf[6]) + " " + str(randomPerf[7]) + "\n")

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            resultFile.write("Net Loss vs Martingale: " + str(n.performance - randomPerf[3]))
            print("=============================================\n")
            resultFile.write("\n\nFinal Predictions: " + str(n.final_prediction))
            resultFile.write("\nFinal Labels: " + str(np.squeeze(n.test_Y)))
            resultFile.write("\n=============================================\n\n")

            """
            AMZN
            """
            print("\nTRAINING AMZN\n")
            tf.reset_default_graph()
            if useSenti:
                n = lstm.LSTM_System(handleData=False, saveModel=True)
            else:
                n = lstm.LSTM_System(handleData=False, saveModel=False)
            n.config.stock = 'AMZN'

            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue = bestParams

            print("Batch Size: ", n.config.batch_size)
            print("Number of Steps: ", n.config.num_steps)
            print("Keep Prob (1-Dropout): ", n.config.keep_prob)
            print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
            print("Number of Hidden Layers: ", n.config.num_layers)
            print("Initial Learning Rate: ", n.config.init_learning_rate)
            print("Williams Period: ", n.config.williams_period)
            print("Average Alpha: ", n.config.avgAlpha)
            print("Learning Rate Decay: ", n.config.learning_rate_decay)
            print("Max epochs: ", n.config.max_epoch)
            print("\n")

            if useSenti:
                n.combineInputs(separated[STOCK])
            else:
                n.df = np.delete(n.df, 0, axis=1)
                n.config.input_size = n.config.input_size - 1
                stats_feat = n.statsFeatures(n.df, n.config)
                n.full = n.normPrice(stats_feat, n.config)
            lstm_graph = n.rnn(config=n.config)
            n.train_lstm(lstm_graph, config=n.config)
            AMZNLoss = n.performance
            AMZNAcc = n.acc_performance

            print("\nAMZN ========================================")
            resultFile.write("\nAMZN ========================================\n")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            resultFile.write("Loss: " + str(n.performance) + " Directional Accuracy: " + str(n.acc_performance) + "\n")
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            resultFile.write("Performance vs Random Agents:\n")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1],
                  "/", randomPerf[2])
            resultFile.write("Loss better than: " + str(randomPerf[0]) + "/" +
                             str(randomPerf[2]) + " Accuracy better than: " + str(randomPerf[1]) + "/" + str(
                randomPerf[2]) + "\n")

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            resultFile.write("Mean and Stdev of Random Loss: " + str(randomPerf[4]) + " " + str(randomPerf[5]) + "\n")
            resultFile.write(
                "Mean and Stdev of Random Accuracy: " + str(randomPerf[6]) + " " + str(randomPerf[7]) + "\n")

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            resultFile.write("Net Loss vs Martingale: " + str(n.performance - randomPerf[3]))
            resultFile.write("\n\nFinal Predictions: " + str(n.final_prediction))
            resultFile.write("\nFinal Labels: " + str(np.squeeze(n.test_Y)))
            print("=============================================\n")
            resultFile.write("\n=============================================\n\n")

            """
            GOOG
            """
            print("\nTRAINING GOOG\n")
            tf.reset_default_graph()
            if useSenti:
                n = lstm.LSTM_System(handleData=False, saveModel=True)
            else:
                n = lstm.LSTM_System(handleData=False, saveModel=False)
            n.config.stock = 'GOOG'

            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue = bestParams

            print("Batch Size: ", n.config.batch_size)
            print("Number of Steps: ", n.config.num_steps)
            print("Keep Prob (1-Dropout): ", n.config.keep_prob)
            print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
            print("Number of Hidden Layers: ", n.config.num_layers)
            print("Initial Learning Rate: ", n.config.init_learning_rate)
            print("Williams Period: ", n.config.williams_period)
            print("Average Alpha: ", n.config.avgAlpha)
            print("Learning Rate Decay: ", n.config.learning_rate_decay)
            print("Max epochs: ", n.config.max_epoch)
            print("\n")

            if useSenti:
                n.combineInputs(separated[STOCK])
            else:
                n.df = np.delete(n.df, 0, axis=1)
                n.config.input_size = n.config.input_size - 1
                stats_feat = n.statsFeatures(n.df, n.config)
                n.full = n.normPrice(stats_feat, n.config)

            lstm_graph = n.rnn(config=n.config)
            n.train_lstm(lstm_graph, config=n.config)
            GOOGLoss = n.performance
            GOOGAcc = n.acc_performance

            print("\nGOOG ========================================")
            resultFile.write("\nGOOG ========================================\n")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            resultFile.write("Loss: " + str(n.performance) + " Directional Accuracy: " + str(n.acc_performance) + "\n")
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            resultFile.write("Performance vs Random Agents:\n")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1],
                  "/", randomPerf[2])
            resultFile.write("Loss better than: " + str(randomPerf[0]) + "/" +
                             str(randomPerf[2]) + " Accuracy better than: " + str(randomPerf[1]) + "/" + str(
                randomPerf[2]) + "\n")

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            resultFile.write("Mean and Stdev of Random Loss: " + str(randomPerf[4]) + " " + str(randomPerf[5]) + "\n")
            resultFile.write(
                "Mean and Stdev of Random Accuracy: " + str(randomPerf[6]) + " " + str(randomPerf[7]) + "\n")

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            resultFile.write("Net Loss vs Martingale: " + str(n.performance - randomPerf[3]))
            resultFile.write("\n\nFinal Predictions: " + str(n.final_prediction))
            resultFile.write("\nFinal Labels: " + str(np.squeeze(n.test_Y)))
            print("=============================================\n")
            resultFile.write("\n=============================================\n\n")

            """
            GOOGL
            """
            print("\nTRAINING GOOGL\n")
            tf.reset_default_graph()
            if useSenti:
                n = lstm.LSTM_System(handleData=False, saveModel=True)
            else:
                n = lstm.LSTM_System(handleData=False, saveModel=False)
            n.config.stock = 'GOOGL'

            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue = bestParams

            print("Batch Size: ", n.config.batch_size)
            print("Number of Steps: ", n.config.num_steps)
            print("Keep Prob (1-Dropout): ", n.config.keep_prob)
            print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
            print("Number of Hidden Layers: ", n.config.num_layers)
            print("Initial Learning Rate: ", n.config.init_learning_rate)
            print("Williams Period: ", n.config.williams_period)
            print("Average Alpha: ", n.config.avgAlpha)
            print("Learning Rate Decay: ", n.config.learning_rate_decay)
            print("Max epochs: ", n.config.max_epoch)
            print("\n")

            if useSenti:
                n.combineInputs(separated[STOCK])
            else:
                n.df = np.delete(n.df, 0, axis=1)
                n.config.input_size = n.config.input_size - 1
                stats_feat = n.statsFeatures(n.df, n.config)
                n.full = n.normPrice(stats_feat, n.config)
            lstm_graph = n.rnn(config=n.config)
            n.train_lstm(lstm_graph, config=n.config)
            GOOGLLoss = n.performance
            GOOGLAcc = n.acc_performance

            print("\nGOOGL ========================================")
            resultFile.write("\nGOOGL ========================================\n")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            resultFile.write("Loss: " + str(n.performance) + " Directional Accuracy: " + str(n.acc_performance) + "\n")
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            resultFile.write("Performance vs Random Agents:\n")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1],
                  "/", randomPerf[2])
            resultFile.write("Loss better than: " + str(randomPerf[0]) + "/" +
                             str(randomPerf[2]) + " Accuracy better than: " + str(randomPerf[1]) + "/" + str(
                randomPerf[2]) + "\n")

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            resultFile.write("Mean and Stdev of Random Loss: " + str(randomPerf[4]) + " " + str(randomPerf[5]) + "\n")
            resultFile.write(
                "Mean and Stdev of Random Accuracy: " + str(randomPerf[6]) + " " + str(randomPerf[7]) + "\n")

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            resultFile.write("Net Loss vs Martingale: " + str(n.performance - randomPerf[3]))
            resultFile.write("\n\nFinal Predictions: " + str(n.final_prediction))
            resultFile.write("\nFinal Labels: " + str(np.squeeze(n.test_Y)))
            print("=============================================\n")
            resultFile.write("\n=============================================\n\n")

            """
            MSFT
            """
            print("\nTRAINING MSFT\n")

            tf.reset_default_graph()
            if useSenti:
                n = lstm.LSTM_System(handleData=False, saveModel=True)
            else:
                n = lstm.LSTM_System(handleData=False, saveModel=False)
            n.config.stock = 'MSFT'

            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, n.config.max_epoch, n.config.early_stopping_continue = bestParams

            print("Batch Size: ", n.config.batch_size)
            print("Number of Steps: ", n.config.num_steps)
            print("Keep Prob (1-Dropout): ", n.config.keep_prob)
            print("Number of Hidden LSTM Cells per Layer: ", n.config.lstm_size)
            print("Number of Hidden Layers: ", n.config.num_layers)
            print("Initial Learning Rate: ", n.config.init_learning_rate)
            print("Williams Period: ", n.config.williams_period)
            print("Average Alpha: ", n.config.avgAlpha)
            print("Learning Rate Decay: ", n.config.learning_rate_decay)
            print("Max epochs: ", n.config.max_epoch)
            print("\n")

            if useSenti:
                n.combineInputs(separated[STOCK])
            else:
                n.df = np.delete(n.df, 0, axis=1)
                n.config.input_size = n.config.input_size - 1
                stats_feat = n.statsFeatures(n.df, n.config)
                n.full = n.normPrice(stats_feat, n.config)
            lstm_graph = n.rnn(config=n.config)
            n.train_lstm(lstm_graph, config=n.config)
            MSFTLoss = n.performance
            MSFTAcc = n.acc_performance

            print("\nMSFT ========================================")
            resultFile.write("\nMSFT ========================================\n")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            resultFile.write("Loss: " + str(n.performance) + " Directional Accuracy: " + str(n.acc_performance) + "\n")
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            resultFile.write("Performance vs Random Agents:\n")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1],
                  "/", randomPerf[2])
            resultFile.write("Loss better than: " + str(randomPerf[0]) + "/" +
                             str(randomPerf[2]) + " Accuracy better than: " + str(randomPerf[1]) + "/" + str(
                randomPerf[2]) + "\n")

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            resultFile.write("Mean and Stdev of Random Loss: " + str(randomPerf[4]) + " " + str(randomPerf[5]) + "\n")
            resultFile.write(
                "Mean and Stdev of Random Accuracy: " + str(randomPerf[6]) + " " + str(randomPerf[7]) + "\n")

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            resultFile.write("Net Loss vs Martingale: " + str(n.performance - randomPerf[3]))
            resultFile.write("\n\nFinal Predictions: " + str(n.final_prediction))
            resultFile.write("\nFinal Labels: " + str(np.squeeze(n.test_Y)))
            print("=============================================\n")
            resultFile.write("\n=============================================\n\n")

            print("Time finished: ", datetime.datetime.now())
    elif trained:
        stockList = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT"]

        for stockName in stockList:
            print("Evaluating for stock", stockName)

            tf.reset_default_graph()

            n = lstm.LSTM_System
            n = lstm.LSTM_System(handleData=False, saveModel=True)
            n.config.stock = stockName
            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(dir_path + '\\..\\..\\resources\\processed\\Pickle_NASDAQ.p')
            n.df = tempDF.loc[:, n.convertStockNames(n._UsedStocks)]
            n.df = n.convertDates(n.df)
            n.df = np.array(n.df[n.config.stock + ' UW Equity', 'BarTp=T'].dropna())

            # IMPORTANT: REMOVE EVERY 2ND ROW
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            n.df = np.delete(n.df, list(range(0, n.df.shape[0], 2)), axis=0)

            n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers, \
            n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay, \
            n.config.max_epoch, n.config.early_stopping_continue = \
                1, 1, 0.49873113702474375, 438, 5, 0.0008495359054701398, 11, 0.493428264464567, 0.9749997977381548, 200, 8

            lstm_graph = n.rnn(config=n.config)
            n.combineInputs(separated[n.config.stock])

            model_dir_path = os.path.dirname(os.path.realpath(__file__))
            model_dir_path += "\\models\\" + n.config.stock + "\\"

            n.restoreAndTest(lstm_graph, model_dir_path + n.config.stock + ".ckpt.meta", model_dir_path)

            AMZNLoss = n.performance
            AMZNAcc = n.acc_performance




            print("\n", n.config.stock,"========================================")
            print("Loss: ", n.performance, " Directional Accuracy: ", n.acc_performance)
            randomPerf = randomAgents.calcPerformance(n, n.performance, n.acc_performance)
            print("Performance vs Random Agents: ")
            print("Loss better than: ", randomPerf[0], "/", randomPerf[2], " Accuracy better than: ", randomPerf[1],
                  "/", randomPerf[2])

            print("Mean and Stdev of Random Loss: ", randomPerf[4], randomPerf[5])
            print("Mean and Stdev of Random Accuracy: ", randomPerf[6], randomPerf[7])

            print("Net Loss vs Martingale: ", n.performance - randomPerf[3])
            print("=============================================\n")
