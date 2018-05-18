import tensorflow as tf
import numpy as np
from dataExtract.dataExtractor import FinExtractor
import os
import pandas as pd
import matplotlib.pyplot as mpl
import random
from sklearn import preprocessing
from sklearn.svm import SVC
from tensorflow.python.client import timeline
from classifier.sentiment.semiSupervised import SemiSupervised
import datetime
import math
from classifier.sentiment import naiveBayes
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score

class LSTM_System:
    #Currently top 5 market cap companies on nasdaq
    _UsedStocks = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN']


    def __init__(self, handleData=True, saveModel=False):
        self.saveModel = saveModel
        self.handleData = handleData
        if(handleData):
            finExtractor = FinExtractor()
            tempDF = finExtractor.FinExtractFromPickle(os.getcwd() + '\\..\\' + finExtractor._pickleL)
            self.df = tempDF.loc[:,self.convertStockNames(self._UsedStocks)]
            self.df = self.convertDates(self.df)
        self.config = config()

    def convertDates(self, df):
        startDate = datetime.datetime.strptime("1900-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        for columns in df.columns.values:
            if not (columns[2] == 'Date'):
                continue
            dates = df.loc[:,columns]
            dateList = []
            for date in dates:
                if(math.isnan(date)):
                    dateList.append(None)
                    continue
                timeDelta = datetime.timedelta(days=date)
                #Fix rounding errors
                if not (timeDelta.microseconds == 0):
                    if(timeDelta.microseconds > 50000):
                        timeDelta = timeDelta + datetime.timedelta(microseconds=1)
                    else:
                        timeDelta = timeDelta - datetime.timedelta(microseconds=1)
                date = startDate + timeDelta
                dateList.append(date)
            df.loc[:,columns] = np.array(dateList)
        return df

    def convertStockNames(self, list):
        return [(s + ' UW Equity') for s in list]


    def rnn(self, config):
        lstm_graph = tf.Graph()
        with lstm_graph.as_default():
            learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

            # Number of examples, number of input, dimension of each input
            inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="inputs")
            targets = tf.placeholder(tf.float32, [None, config.output_size], name="targets")

            def _create_one_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
                if config.keep_prob < 1.0:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
                return lstm_cell

            if (config.num_layers > 1):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [_create_one_cell() for _ in range(config.num_layers)],
                    state_is_tuple=True
                )
            else:
                cell = _create_one_cell()

            init_state = tf.placeholder(tf.float32, [config.num_layers, 2, config.batch_size, config.lstm_size], name="init_state")
            state_per_layer_list = tf.unstack(init_state, axis=0)

            if(config.num_layers == 1):
                rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[0][0], state_per_layer_list[0][1])])
            else:
                rnn_tuple_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[i][0], state_per_layer_list[i][1])
                     for i in range(config.num_layers)]
                )

            """
            TODO:
            ValueError, not enough values to unpack for num_layers = 1
            """
            outputs, self.current_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

            outputs = tf.transpose(outputs, [1,0,2])
            print("outputs shape: ", outputs.get_shape)

            with tf.name_scope("output_layer"):

                weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.output_size]), name="weights")

                #initialize bias as 1, as suggested in Jozefowicz, Zaremba, Sutskever (2015)
                bias = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[config.output_size]), name="biases")
                prediction = tf.matmul(outputs[-1], weight) + bias

                tf.summary.histogram("last_lstm_output", outputs[-1])
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)

            with tf.name_scope("train"):
                if(config.huber):
                    loss = tf.reduce_mean(tf.losses.huber_loss(labels=targets, predictions=prediction), name="loss_huber")
                else:
                    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=targets, predictions=prediction), name="loss_huber")
                optimizer = tf.train.AdamOptimizer(learning_rate)
                minimize = optimizer.minimize(loss, name="loss_huber_adam_minimize")
                tf.summary.scalar("loss_huber", loss)



            # Operators to use after restoring the model
            for op in [prediction, loss]:
                tf.add_to_collection('ops_to_restore', op)

        return lstm_graph


    #Normalize by taking the percentage change from last time step
    def normPrice(self, input, config):
        diff = np.array(input)

        if(input.ndim == 1):

            train_size = int(len(diff) * (config.test_train_split))
            self.mean = np.mean(diff[:train_size], axis=0)
            self.stdev = np.std(diff[:train_size])
            diff = diff - self.mean
            diff = diff / self.stdev
            return np.concatenate((np.array([0]),diff))
        elif(input.ndim == 2):
            print("input ndim 2")


            train_size = int(diff.shape[0] * (config.test_train_split))
            self.testPrices = diff[int((diff.shape[0]-config.num_steps)*config.test_train_split)+config.num_steps:,0]


            last = diff[0,0]
            for i in range(diff.shape[0]):
                templast = diff[i,0]
                diff[i,0] = diff[i,0]/last
                last = templast


            self.scaler = preprocessing.StandardScaler()
            transformed = diff[:train_size, 0].reshape(-1,1)
            self.scaler.fit(transformed)

            self.fullScaler = preprocessing.StandardScaler()
            self.fullScaler.fit(diff[:train_size])
            return self.fullScaler.transform(diff)


    def prepare_inputs(self, config, input, labels):

        # split into items of input_size
        print("Input Dimensions: ", input.ndim)
        if(input.ndim == 2):
            input = np.squeeze([input[i: (i + 1)]
                     for i in range(len(input))])
            labels = np.squeeze([labels[i: (i + 1)]
                   for i in range(len(labels))])

            X = np.array([input[i: i + config.num_steps] for i in range(len(input) - config.num_steps)])
            y = np.array([labels[i + config.num_steps] for i in range(len(labels) - config.num_steps)])
            train_size = int(len(X) * (config.test_train_split))
            train_X, test_X = X[:train_size], X[train_size:]
            train_y, test_y = np.reshape(y[:train_size,0], [-1,1]), np.reshape(y[train_size:,0], [-1,1])
            return train_X, train_y, test_X, test_y
        else:
            input = [np.array(input[i * config.input_size: (i + 1) * config.input_size])
                     for i in range(len(input) // config.input_size)]
            labels = [np.array(labels[i * config.input_size: (i + 1) * config.input_size])
                      for i in range(len(labels) // config.input_size)]

            X = np.array([input[i: i + config.num_steps] for i in range(len(input) - config.num_steps)])
            y = np.array([labels[i + config.num_steps] for i in range(len(labels) - config.num_steps)])

            train_size = int(len(X) * (config.test_train_split))
            train_X, test_X = X[:train_size], X[train_size:]
            train_y, test_y = y[:train_size], y[train_size:]
            return train_X, train_y, test_X, test_y



    def generate_one_epoch(self, batch_size, train=True):
        if(train):
            num_batches = int(len(self.train_X)) // batch_size
            if batch_size * num_batches < len(self.train_X):
                num_batches += 1
        else:
            num_batches = int(len(self.test_X)) // batch_size
            if batch_size * num_batches < len(self.test_X):
                num_batches += 1
        batch_indices = list(range(num_batches))

        for j in batch_indices:
            if (train):
                batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
                batch_y = self.train_Y[j * batch_size: (j + 1) * batch_size]
            else:
                if (batch_size > 1):
                    yield self.test_X, self.test_Y
                else:
                    batch_X = self.test_X[j * batch_size: (j + 1) * batch_size]
                    batch_y = self.test_Y[j * batch_size: (j + 1) * batch_size]
            yield batch_X, batch_y


    def _compute_learning_rates(self, config):
        learning_rates_to_use = [
            config.init_learning_rate * (
                config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
            ) for i in range(config.max_epoch)
        ]
        return learning_rates_to_use

    def shapeSentiFeats(self, dates, sentiFeats):
        dates.reshape(-1, 1)
        zeros = np.zeros(shape=dates.shape)
        dates = np.concatenate((dates, zeros), axis=1)
        for feat in sentiFeats:
            for i in range(0, len(dates)):
                if(dates[i,0] > feat[0]):
                    dates[i,1] += feat[1]
                    break

        return dates[:,1].reshape(-1,1)

    def combineInputs(self, senti_separated):
        if not (self.config.train_validation_split == 0):
            train_size = int(len(self.df) * (self.config.test_train_split))
            self.df = self.df[0:train_size]
        if not (config.sin_data):
            if(self.handleData):
                df = self.df[config.stock + ' UW Equity', 'BarTp=T'].dropna()
                df = np.array(df)
            else:
                df = self.df
            deleted = df[:,0].reshape(-1, 1)

            sentiFeats = self.shapeSentiFeats(deleted, senti_separated)

            df = np.delete(df, 0, axis=1)


            stats_feat = self.statsFeatures(df, config)
            self.full = self.normPrice(stats_feat, self.config)

            print("df shape, sentifeats shape", self.full.shape, sentiFeats.shape)


            self.full = np.concatenate((self.full, sentiFeats), axis=1)

        else:
            sinD = np.delete(np.array(sinData(1000)), 0, axis=1)
            stats_feat = self.statsFeatures(sinD, self.config)
            self.full = self.normPrice(stats_feat, self.config)

        return self.full

    def train_lstm(self, lstm_graph, config):

        self.train_X, self.train_Y, self.test_X, self.test_Y = \
            self.prepare_inputs(config, np.array(self.full[:-1]), np.array(self.full[1:]))

        graph_name = "%s_lr%.2f_lr_decay%.3f_lstm%d_step%d_input%d_batch%d_epoch%d" % (
            'MSFT',
            config.init_learning_rate, config.learning_rate_decay,
            config.lstm_size, config.num_steps,
            config.input_size, config.batch_size, config.max_epoch)
        learning_rates_to_use = self._compute_learning_rates(config)
        with tf.Session(graph=lstm_graph) as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('_logs/' + graph_name, sess.graph)
            writer.add_graph(sess.graph)

            graph = tf.get_default_graph()
            tf.global_variables_initializer().run()

            inputs = graph.get_tensor_by_name('inputs:0')
            targets = graph.get_tensor_by_name('targets:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')
            init_state = graph.get_tensor_by_name('init_state:0')

            loss = graph.get_tensor_by_name('train/loss_huber:0')
            minimize = graph.get_operation_by_name('train/loss_huber_adam_minimize')
            prediction = graph.get_tensor_by_name('output_layer/add:0')

            min_valid_loss = np.inf
            stopping_counter = 0

            end_epoch = config.max_epoch
            print("train mean and stdev =", np.mean(self.train_Y[:,0]), np.std(self.train_Y[:,0]))
            print('test mean and stdev =', np.mean(self.test_Y[:,0]), np.std(self.test_Y[:,0]))

            for epoch_step in range(config.max_epoch):
                current_lr = learning_rates_to_use[epoch_step]
                train_losses = []
                train_accuracies = []

                # Reset initial state between epochs
                _current_state = np.zeros((config.num_layers, 2, config.batch_size, config.lstm_size))


                for batch_X, batch_y in self.generate_one_epoch(config.batch_size):
                    train_data_feed = {
                        inputs: batch_X,
                        targets: batch_y,
                        learning_rate: current_lr,
                        init_state: _current_state
                    }

                    train_loss, _, _current_state, train_pred = sess.run([loss, minimize, self.current_state, prediction], train_data_feed)

                    train_losses.append(train_loss)
                    train_perf = self.calcPerformance(batch_y, train_pred, config, batch_X, train=True)
                    train_accuracies.append(float(train_perf[0])/float(train_perf[1]))

                test_losses = []
                test_accuracies = []
                test_preds = []
                for batch_X, batch_y in self.generate_one_epoch(config.batch_size, train=False):
                    test_data_feed = {
                        inputs: batch_X,
                        targets: batch_y,
                        learning_rate: 0.0,
                        init_state: _current_state
                    }

                    test_loss, _pred, _summary, _current_state = sess.run(
                        [loss, prediction, merged_summary, self.current_state], test_data_feed)
                    test_preds.append(_pred)
                    test_losses.append(test_loss)
                    test_perf = self.calcPerformance(batch_y, _pred, config, batch_X, train=False)
                    test_accuracies.append(float(test_perf[0]) / float(test_perf[1]))

                if epoch_step % 5 == 0:

                    assert len(test_preds) == len(self.test_Y)
                    print("Epoch %d [%f]:" % (epoch_step, current_lr), "mean test loss: ", np.mean(test_losses),
                          "mean train loss: ", np.mean(train_losses), "mean train accuracy: ", np.mean(train_accuracies))
                    print("Directional accuracy of predictions")
                    print("NumCorrect: ", int(round(self.test_Y.shape[0]*np.mean(test_accuracies))))
                    print("Total: ", self.test_Y.shape[0])
                    print("Percentage: ", np.mean(test_accuracies))

                writer.add_summary(_summary, global_step=epoch_step)
                """
                Early Stopping
                """

                # if(np.mean(test_accuracies) > 0.7):
                #     print("Early Stopping from Significant Accuracy")
                #     end_epoch = epoch_step
                #     break

                if(np.mean(test_losses) < min_valid_loss):
                    stopping_counter = 0
                    min_valid_loss = np.mean(test_losses)
                elif(np.mean(test_losses) < min_valid_loss + config.early_stopping_leeway):
                    stopping_counter = 0
                else:
                    stopping_counter += 1
                    #print("Possible Early Stopping: Epoch ", epoch_step, ". Min: ", min_valid_loss, ". Current: ", test_loss)
                if(stopping_counter >= config.early_stopping_continue):
                    print("Early Stopping Activated")
                    end_epoch = epoch_step
                    break
            self.final_prediction = np.squeeze(test_preds)
            print("Final Loss: ", np.mean(test_losses))
            self.final_loss = test_loss

            timesteps = range(len(self.final_prediction))

            mpl.plot(timesteps, self.test_Y, label='data')

            mpl.plot(timesteps, self.final_prediction, label='predict')
            mpl.xlabel('Time')
            mpl.ylabel('Normalized price change')
            mpl.legend()
            fig1 = mpl.gcf()

            if(self.saveModel):
                dir_path = os.path.dirname(os.path.realpath(__file__))
                dir_path += "\\models\\" + self.config.stock + "\\"
                fig1.savefig(dir_path + self.config.stock + str(self.config.init_learning_rate) + "_" + str(end_epoch) + "_" +
                         str(np.mean(test_accuracies)) + "_" + str(self.config.williams_period) + "_" +
                         str(self.config.avgAlpha) + "_" + str(self.config.learning_rate_decay)
                         + "_20min.png", dpi=1200)
            mpl.gcf().clear()
            mpl.close()

            print("Directional accuracy of predictions")
            print("NumCorrect: ", self.test_Y.shape[0] * np.mean(test_accuracies))
            print("Total: ", self.test_Y.shape[0])
            print("Percentage: ", np.mean(test_accuracies))

            self.acc_performance = np.mean(test_accuracies)

            self.performance = np.mean(test_losses)

            if(self.saveModel):
                dir_path = os.path.dirname(os.path.realpath(__file__))
                dir_path += "\\models\\" + self.config.stock + "\\"
                saver = tf.train.Saver()
                saver.save(sess, dir_path + self.config.stock + ".ckpt")

    def calcPerformance(self, actual, predicted, config, first, train=False):

        assert (predicted.shape == actual.shape)

        totalPreds = len(predicted)
        numCorrect = 0


        zeros = np.zeros((predicted.shape[0], config.input_size))
        zeros[:,0] = np.reshape(predicted, (-1))
        predicted = zeros

        newZeros = np.zeros((actual.shape[0], config.input_size))
        newZeros[:,0] = np.reshape(actual, (-1))
        actual = newZeros


        predicted = predicted[:,0]
        actual = actual[:,0]


        for i in range(0, len(actual)):
            last = first[i, 0, 0]
            if(config.sin_data == True):
                if(np.sign(predicted[i]) == np.sign(actual[i])):
                    numCorrect += 1
            else:
                if(np.sign(predicted[i] - last) == np.sign(actual[i] - last)):
                    numCorrect += 1


        """
        MSE
        """
        MSE = np.square(predicted - actual).mean()

        return numCorrect, totalPreds, MSE

    def statsFeatures(self, input, config):
        slow_williamsR = []
        RoC = []
        momentum = []
        fast_williamsR = []
        RSI = []
        TEMA = []

        upChangeAvg = 0
        downChangeAvg = 0


        lastPrice = input[0,0]
        priceAvg = input[0,0]

        ema_ema = input[0,0]
        ema_ema_ema = input[0,0]
        for i in range(0, input.shape[0]):

            """
            SLOW WILLIAMS
            """
            if(i == 0):
                highest = input[0,0]
                lowest = input[0,0]
                slow_williamsR.append(np.float32(-0.5))
            elif(i <= config.williams_period):
                highest = np.amax(input[:i+1,0])
                lowest = np.amin(input[:i+1,0])
                slow_williamsR.append(-1. * (highest - input[i, 0]) / (highest - lowest))
            else:
                highest = np.amax(input[i-config.williams_period:i+1,0])
                lowest = np.amin(input[i-config.williams_period:i+1,0])
                slow_williamsR.append(-1.*(highest-input[i,0])/(highest-lowest))

            """
            ROC AND MOMENTUM
            """

            if(i == 0):
                RoC.append(np.float32(0))
                momentum.append(np.float32(0))
            else:
                RoC.append(input[i,0]/input[i-1,0] - 1)
                momentum.append(input[i,0]-input[i-1,0])

            """
            FAST WILLIAMS
            """
            if (i == 0):
                highest = input[0, 0]
                lowest = input[0, 0]
                fast_williamsR.append(np.float32(-0.5))
            elif (i <= (config.williams_period/2)):
                highest = np.amax(input[:i+1, 0])
                lowest = np.amin(input[:i+1, 0])
                fast_williamsR.append(-1. * (highest - input[i, 0]) / (highest - lowest))
            else:
                highest = np.amax(input[i - int(config.williams_period/2):i+1, 0])
                lowest = np.amin(input[i - int(config.williams_period/2):i+1, 0])
                fast_williamsR.append(-1. * (highest - input[i, 0]) / (highest - lowest))

            if(input[i,0] > lastPrice):
                upChangeAvg = (config.avgAlpha*(input[i,0]-lastPrice)) + (1-config.avgAlpha)*upChangeAvg
                downChangeAvg = (1-config.avgAlpha)*downChangeAvg
            else:
                downChangeAvg = (config.avgAlpha * (lastPrice - input[i, 0])) + (1 - config.avgAlpha) * downChangeAvg
                upChangeAvg = (1 - config.avgAlpha) * upChangeAvg
            if(downChangeAvg == 0):
                RS = 2
            else:
                RS = upChangeAvg/downChangeAvg
            RSI_calc = 100 - 100./(1+RS)
            RSI.append(RSI_calc)

            priceAvg = (config.avgAlpha*input[i,0]) + (1-config.avgAlpha)*priceAvg
            ema_ema = (config.avgAlpha*priceAvg) + (1-config.avgAlpha)*ema_ema
            ema_ema_ema = (config.avgAlpha*ema_ema) + (1-config.avgAlpha)*ema_ema_ema

            TEMA_calc = 3*priceAvg - 3*ema_ema + ema_ema_ema
            TEMA.append(TEMA_calc)

            lastPrice = input[i,0]

        #Currently hard-coded in input
        allFeatures = np.array(list(zip(input[:,0], input[:,1], slow_williamsR, RoC, momentum, fast_williamsR, RSI, TEMA)))

        # return allFeatures
        return allFeatures

    def restoreAndTest(self, lstm_graph, location, directory):

        self.train_X, self.train_Y, self.test_X, self.test_Y = \
            self.prepare_inputs(self.config, np.array(self.full[:-1]), np.array(self.full[1:]))

        new_graph = tf.Graph()

        with tf.Session(graph=new_graph) as sess:
            saver = tf.train.import_meta_graph(location)
            saver.restore(sess, tf.train.latest_checkpoint(directory))

            graph = tf.get_default_graph()


            inputs = graph.get_tensor_by_name('inputs:0')
            targets = graph.get_tensor_by_name('targets:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')
            init_state = graph.get_tensor_by_name('init_state:0')

            """
            Way to try to get unnamed state variable to hook
            """
            tempTensor = []
            for i in range(1, self.config.num_layers + 1):
                odd = i * 2 + 1
                even = i * 2 + 2
                tempTensor.append(tf.nn.rnn_cell.LSTMStateTuple(graph.get_tensor_by_name('rnn/while/Exit_' + str(odd) + ':0'), graph.get_tensor_by_name('rnn/while/Exit_' + str(even) + ':0')))

            self.current_state = tuple(tempTensor)

            loss = graph.get_tensor_by_name('train/loss_huber:0')

            prediction = graph.get_tensor_by_name('output_layer/add:0')

            train_losses = []
            train_accuracies = []

            # Reset initial state between epochs
            _current_state = np.zeros((self.config.num_layers, 2, self.config.batch_size, self.config.lstm_size))

            for batch_X, batch_y in self.generate_one_epoch(self.config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: 0.0,
                    init_state: _current_state
                }

                train_loss, _current_state, train_pred = sess.run([loss, self.current_state, prediction],
                                                                     train_data_feed)

                train_losses.append(train_loss)
                train_perf = self.calcPerformance(batch_y, train_pred, self.config, batch_X, train=True)
                train_accuracies.append(float(train_perf[0]) / float(train_perf[1]))

            test_losses = []
            test_accuracies = []
            test_preds = []
            counter = 0
            for batch_X, batch_y in self.generate_one_epoch(self.config.batch_size, train=False):
                test_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: 0.0,
                    init_state: _current_state
                }

                test_loss, _pred, _current_state = sess.run(
                    [loss, prediction, self.current_state], test_data_feed)
                test_preds.append(_pred)
                test_losses.append(test_loss)
                test_perf = self.calcPerformance(batch_y, _pred, config, batch_X, train=False)
                test_accuracies.append(float(test_perf[0]) / float(test_perf[1]))
                if(counter % 50 == 0):
                    print("Example", counter, "of", len(self.test_Y))
                    print("Current MSE", np.mean(test_losses))
                    print("Current Accuracy", np.mean(test_accuracies))
                    print("========================================")
                counter += 1

            assert len(test_preds) == len(self.test_Y)

            print("R^2 Value: ", r2_score(np.squeeze(self.test_Y), np.squeeze(test_preds)))

            print("Test set evaluation complete")

            self.acc_performance = np.mean(test_accuracies)

            self.performance = np.mean(test_losses)



class config:
    keep_prob = 1#0.85
    input_size = 9
    output_size = 1
    batch_size = 512
    init_learning_rate = 0.001
    learning_rate_decay = 0.95
    lstm_size = 185
    num_layers = 1
    num_steps = 1
    test_train_split = 0.8
    MODEL_DIR = os.getcwd() + "\\models"
    init_epoch = 5

    max_epoch = 2000
    train_validation_split = 0 # Set to 0 if not running validation
    early_stopping_continue = 100
    huber = False

    early_stopping_leeway = 0.01

    #TESTING
    sin_data = False


    #FEATURES
    williams_period = 10
    avgAlpha = 0.2
    stock = "AAPL"

def sinData(numSteps):
    time = np.arange(numSteps)
    print(np.sin(np.pi * (time + 0.5)) + np.random.normal(0,0.1,size=(len(time))))
    list = (np.sin(np.pi * (time + 0.5)) + np.random.normal(0,0.1,size=(len(time))))
    someNum = np.zeros(shape=list.shape) + np.random.normal(0,0.0001,size=(len(time)))

    stacked = np.stack((time, list, someNum), axis=-1)
    return pd.DataFrame(stacked)

def getAndSepSentimentData():
    """
    deprecated
    :return:
    """
    """
        Sentiment Features

        """
    ss = SemiSupervised()
    dates, data, labels = ss.getTwitterRaw('../../resources/hydrated_tweets/relevant_tweets.txt')
    labels = [int(label) for label in labels]

    trainPercent = config.test_train_split

    trainDates = dates[0:int(trainPercent * len(dates))]
    trainData = data[0:int(trainPercent * len(data))]
    trainLabels = labels[0:int(trainPercent * len(labels))]
    assert (len(trainData) == len(trainLabels))
    assert (len(trainDates) == len(trainData))

    structure = naiveBayes.BoWStruct(trainData)
    FV = []
    for sentence, value in zip(data, labels):
        FV.append(naiveBayes.getVector(sentence, structure))
    FV = np.array(FV)
    trainFV = FV[0:int(trainPercent * len(FV))]
    assert (len(trainFV) == len(trainLabels))

    testDates = dates[int(trainPercent * len(dates)):]
    testData = data[int(trainPercent * len(data)):]
    testLabels = labels[int(trainPercent * len(labels)):]
    assert (len(testData) == len(testLabels))
    assert (len(testDates) == len(testData))
    testFV = FV[int(trainPercent * len(FV)):]
    assert (len(testFV) == len(testLabels))

    SVM = SVC(C=10, gamma=0.05, kernel='rbf')
    SVM.fit(trainFV, trainLabels)

    predLabels = SVM.predict(FV)
    assert(len(predLabels) == len(labels))

    separated = {}
    stockNames = ['apple', 'google', 'alphabet', 'microsoft', 'amazon']
    for name in stockNames:
        separated[name] = []



    for name in stockNames:
        for dateTime, _data, label in zip(dates, data, predLabels):
            if (name in _data):
                separated[name].append([dateTime, label])

    return separated

if __name__ == "__main__":
    tf.reset_default_graph()

    sep = getAndSepSentimentData()

    """
    Hyper parameter tuning
    """

    errorDict = {}
    #Random search
    for n in range(0, 30):
        n = LSTM_System(sep)
        n.config.num_layers = 3 #np.random.choice(3, p=[0.5, 0.25, 0.25]) + 2
        n.config.batch_size = 1 #np.random.randint(1,300/n.config.num_layers)
        #n.config.num_steps = np.random.randint(1,n.config.batch_size+1)
        n.config.keep_prob = np.random.uniform(0.3,0.8)
        n.config.lstm_size = np.random.randint(512, 2048)
        n.config.init_learning_rate = np.random.uniform(0.0005,0.001)
        n.config.max_epoch = int(float(2000)/n.config.num_layers)
        n.config.early_stopping_continue = int(float(200)/n.config.num_layers)
        n.config.williams_period = np.random.randint(10,20)
        n.config.avgAlpha = np.random.uniform(0.1,0.5)
        n.config.learning_rate_decay = np.random.uniform(0.96,0.99)
        #Currently just testing microsoft
        #n.config.stock = n._UsedStocks[np.random.randint(0,len(n._UsedStocks))]


        print("\n\n", n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers,
              n.config.init_learning_rate, n.config.williams_period, n.config.avgAlpha, n.config.learning_rate_decay)
        lstm_graph = n.rnn(config=n.config)
        n.train_lstm(lstm_graph, config=n.config)
        errorDict[n.performance] = str(n.config.batch_size)+","+str(n.config.num_steps)+","+str(n.config.keep_prob)+","+\
                                  str(n.config.lstm_size)+","+str(n.config.num_layers)+","+str(n.config.init_learning_rate)\
                                   +","+str(n.config.williams_period)+","+str(n.config.avgAlpha)+","+str(n.config.learning_rate_decay)
        print(n.final_loss)
        print("\nDONE\n", n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers,
              n.config.init_learning_rate, n.config.williams_period,n.config.avgAlpha)
    keyList = (sorted(list(errorDict.keys()), reverse=True))
    for i in range(0, len(keyList)):
        print(errorDict[keyList[i]], "Directional Accuracy: ", keyList[i])
    print(np.mean(keyList))
    print(np.std(keyList))


    # n = LSTM_System()
    # lstm_graph = n.rnn(config=n.config)
    # n.train_lstm(lstm_graph, config=n.config)

    #Comprehensive search
    """
    for n in range(10,20):
        dropout = float(n)/20
        for i in range(1,10):
            batch_size = i*40 - 30
            for j in range(1, 5):
                step_size = int(float(j*batch_size)/5)
                n = LSTM_System()
                n.config.batch_size = batch_size
                n.config.num_steps = step_size
                n.config.keep_prob
                print("\n\n", batch_size, step_size)
                lstm_graph = n.rnn(config=n.config)
                n.train_lstm(lstm_graph, config=n.config)
                errorDict[n.final_loss] = str(dropout)+","+str(batch_size)+","+str(step_size)
                print("DONE\n")
    """
