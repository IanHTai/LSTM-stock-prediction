import tensorflow as tf
import numpy as np
from dataExtract.dataExtractor import FinExtractor
import os
import pandas as pd
import matplotlib.pyplot as mpl
import random
from sklearn import preprocessing

class LSTM_System:
    #Currently top 5 market cap companies on nasdaq
    _UsedStocks = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN']


    def __init__(self):
        finExtractor = FinExtractor()
        tempDF = finExtractor.FinExtractFromPickle(os.getcwd() + '\\..\\' + finExtractor._pickleL)
        print('LSTM Initialized')
        self.df = tempDF[self.convertStockNames(self._UsedStocks)]
        #print(self.df)
        self.config = config()
        #self.full = self.normPrice(self.df['AAPL UW Equity', 'BarTp=T', 'OPEN'].dropna(), self.config)


    def convertStockNames(self, list):
        return [(s + ' UW Equity') for s in list]


    # def get_state_update(self, stateVar, newStates):
    #     # Operation to update state with last state's tensors
    #     updateOps = []
    #     print(stateVar)
    #     for state_variable, new_state in zip(stateVar, newStates):
    #         updateOps.extend([state_variable[0].assign(new_state[0]), state_variable[1].assign(new_state[1])])
    #
    #     return tf.tuple(updateOps)
    #
    # def set_state_zeros(self, states):
    #     updateOps = []
    #     for state_variable in states:
    #         updateOps.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
    #                            state_variable[1].assign(tf.zeros_like(state_variable[1]))])
    #
    #     return tf.tuple(updateOps)

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
                print(rnn_tuple_state)
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
                # last.get_shape() = (batch_size, lstm_size)
                #last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_lstm_output")

                weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.output_size]), name="weights")

                #initialize bias as 1, as suggested in Jozefowicz, Zaremba, Sutskever (2015)
                bias = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[config.output_size]), name="biases")
                prediction = tf.matmul(outputs[-1], weight) + bias

                tf.summary.histogram("last_lstm_output", outputs[-1])
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)

            with tf.name_scope("train"):
                # loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
                #loss = tf.reduce_mean(tf.square(prediction - targets), name="loss_mse")
                loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=targets, predictions=prediction), name="loss_mse")
                optimizer = tf.train.AdamOptimizer(learning_rate)
                minimize = optimizer.minimize(loss, name="loss_mse_adam_minimize")
                tf.summary.scalar("loss_mse", loss)



            # Operators to use after restoring the model
            for op in [prediction, loss]:
                tf.add_to_collection('ops_to_restore', op)

        return lstm_graph


    # def serving_input_fn(self):
    #     feature_placeholders = {
    #         self.TIMESERIES_COL: tf.placeholder(tf.float32, [None, self.config.input_size])
    #     }
    #
    #     features = {
    #         key: tf.expand_dims(tensor, -1)
    #         for key, tensor in feature_placeholders.items()
    #     }
    #     features[self.TIMESERIES_COL] = tf.squeeze(features[self.TIMESERIES_COL], axis=[2])
    #
    #     print
    #     'serving: features={}'.format(features[self.TIMESERIES_COL])
    #
    #     return tflearn.utils.input_fn_utils.InputFnOps(
    #         features,
    #         None,
    #         feature_placeholders
    #     )

    # from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
    # def experiment(self, output_dir):
    #     return tflearn.Experiment(tflearn.Estimator(model_fn=self.rnn, model_dir=output_dir), train_input_fn=self.input_fn(), eval_input_fn=self.test_fn(), eval_metrics={
    #         'rmse': tflearn.MetricSpec(
    #             metric_fn=metrics.streaming_root_mean_squared_error
    #         )
    #     },
    #                               export_strategies=[saved_model_export_utils.make_export_strategy(
    #                                   self.serving_input_fn,
    #                                   default_output_alternative_key=None,
    #                                   exports_to_keep=1
    #                               )]
    #                               )

    #Normalize by taking the percentage change from last time step
    def normPrice(self, input, config):
        diff = np.array(input)
        if not (config.train_validation_split == 0):
            train_size = int(len(diff) * (config.test_train_split))
            diff = diff[0:train_size]


        if(input.ndim == 1):
            #diff = np.divide(np.diff(input), input[:-1])

            train_size = int(len(diff) * (config.test_train_split))
            self.mean = np.mean(diff[:train_size])
            self.stdev = np.std(diff[:train_size])
            diff = diff - self.mean
            diff = diff / self.stdev
            #print(diff)
            return np.concatenate((np.array([0]),diff))
        elif(input.ndim == 2):
            print("input ndim 2")

            # diff = np.divide(np.diff(diff[:,0]), diff[:-1,0])

            train_size = int(diff.shape[0] * (config.test_train_split))
            # print("train length", diff.shape[0])
            # print("train_size with 1 taken into account", int((diff.shape[0]+1)*config.test_train_split))

            self.testPrices = diff[int((diff.shape[0]-config.num_steps)*config.test_train_split)+config.num_steps:,0]


            # np.savetxt("temp/nonPercentFeatures.csv", diff, delimiter=',')


            last = diff[0,0]
            for i in range(diff.shape[0]):
                templast = diff[i,0]
                diff[i,0] = diff[i,0]/last
                last = templast

            # self.mean = np.mean(diff[:train_size], axis=0)
            #
            # ###########
            # ###########
            # # self.mean[0] = 1
            # ###########
            # ###########
            # self.stdev = np.std(diff[:train_size], axis=0)
            # diff = diff - self.mean
            # diff = diff / self.stdev

            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(diff[:train_size])


            return self.scaler.transform(diff)


    def prepare_inputs(self, config, input, labels):

        # split into items of input_size
        print("Input Dimensions: ", input.ndim)
        if(input.ndim == 2):
            input = np.squeeze([input[i: (i + 1)]
                     for i in range(len(input))])
            #print(np.array(input).shape)
            labels = np.squeeze([labels[i: (i + 1)]
                   for i in range(len(labels))])

            X = np.array([input[i: i + config.num_steps] for i in range(len(input) - config.num_steps)])
            y = np.array([labels[i + config.num_steps] for i in range(len(labels) - config.num_steps)])
            # print("input length", input.shape[0])
            # print("x len", len(X))
            train_size = int(len(X) * (config.test_train_split))
            train_X, test_X = X[:train_size], X[train_size:]
            train_y, test_y = np.reshape(y[:train_size,0], [-1,1]), np.reshape(y[train_size:,0], [-1,1])
            # print("train_x length", len(train_X))
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

    # def _input_fn(self):
    #     length = int(self.config.test_train_split * self.full.size)
    #     input = np.array(self.full[:length - 1], dtype='float32')
    #     labels = np.array(self.full[1:length], dtype='float32')

        # self.train_X = input
        # self.train_Y = labels
        # return input, labels

    # def _test_fn(self):
    #     length = int(1 - self.config.test_train_split) * self.full.size
    #     input = np.array(self.full[length + 1:-1], dtype='float32')
    #     labels = np.array(self.full[length + 2:], dtype='float32')
    #     self.test_X = input
    #     self.test_Y = labels
    #     return input, labels


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

        # random.shuffle(batch_indices)
        for j in batch_indices:
            if (train):
                batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
                batch_y = self.train_Y[j * batch_size: (j + 1) * batch_size]
            else:
                batch_X = self.test_X[j * batch_size: (j + 1) * batch_size]
                batch_y = self.test_Y[j * batch_size: (j + 1) * batch_size]
            #assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y


    def _compute_learning_rates(self, config):
        learning_rates_to_use = [
            config.init_learning_rate * (
                config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
            ) for i in range(config.max_epoch)
        ]
        #print("Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2])
        return learning_rates_to_use

    def train_lstm(self, lstm_graphm, config):
        #Normalize full to mean = 0 and stdev = 1
        #NO LONGER NEEDED
        #self.full = np.array(self.full)
        #self.full = self.full - np.mean(self.full)
        #self.full = self.full / np.std(self.full)
        if not (config.sin_data):
            df = self.df[config.stock + ' UW Equity', 'BarTp=T'].dropna()
            df = np.delete(np.array(df), 0, axis=1)

            # IMPORTANT: REMOVE EVERY 2ND ROW
            df = np.delete(df, list(range(0, df.shape[0], 2)), axis=0)

            # REMOVE EVERY 2ND ROW AGAIN
            df = np.delete(df, list(range(0, df.shape[0], 2)), axis=0)


            stats_feat = self.statsFeatures(df, config)
            self.full = self.normPrice(stats_feat, self.config)
        else:
            sinD = np.delete(np.array(sinData(8893)), 0, axis=1)
            stats_feat = self.statsFeatures(sinD, self.config)
            self.full = self.normPrice(stats_feat, self.config)
            # print(self.full)
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

            loss = graph.get_tensor_by_name('train/loss_mse:0')
            minimize = graph.get_operation_by_name('train/loss_mse_adam_minimize')
            prediction = graph.get_tensor_by_name('output_layer/add:0')

            min_valid_loss = np.inf
            stopping_counter = 0

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
                    train_perf = self.calcPerformance(batch_y, train_pred, train=True)
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

                    test_loss, _pred, _summary, test_state = sess.run([loss, prediction, merged_summary, self.current_state], test_data_feed)
                    test_preds.append(_pred)
                    test_losses.append(test_loss)
                    test_perf = self.calcPerformance(batch_y, _pred, train=False)
                    test_accuracies.append(float(test_perf[0])/float(test_perf[1]))

                assert len(test_preds) == len(self.test_Y)
                if epoch_step % 5 == 0:
                    print("Epoch %d [%f]:" % (epoch_step, current_lr), "mean test loss: ", np.mean(test_losses),
                          "mean train loss: ", np.mean(train_losses), "mean train accuracy: ", np.mean(train_accuracies))
                    if epoch_step% 10 == 0:
                        print("Directional accuracy of predictions")
                        print("NumCorrect: ", self.test_Y.shape[0]*np.mean(test_accuracies))
                        print("Total: ", self.test_Y.shape[0])
                        print("Percentage: ", np.mean(test_accuracies))

                        # print("Backtest starting with $10000", self.backTest(self.testPrices, _pred))
                    if epoch_step % 50 == 0:
                        """
                        print("Predictions:", [(
                            list(map(lambda x: round(x, 4), _pred[-j])),
                            list(map(lambda x: round(x, 4), self.test_Y[-j]))
                        ) for j in range(5)])
                        
                        
                        """

                writer.add_summary(_summary, global_step=epoch_step)
                """
                Early Stopping
                """
                if(test_loss < min_valid_loss):
                    stopping_counter = 0
                    min_valid_loss = test_loss
                else:
                    stopping_counter += 1
                    #print("Possible Early Stopping: Epoch ", epoch_step, ". Min: ", min_valid_loss, ". Current: ", test_loss)
                if(stopping_counter >= config.early_stopping_continue):
                    print("Early Stopping Activated")
                    break
            final_prediction = _pred
            print("Final Loss: ", test_loss)
            self.final_loss = test_loss

            timesteps = range(len(final_prediction))

            mpl.plot(timesteps, self.test_Y, label='data')
            mpl.plot(timesteps, final_prediction, label='predict')
            mpl.xlabel('Time')
            mpl.ylabel('Normalized price change')
            mpl.legend()
            fig1 = mpl.gcf()
            # mpl.show()
            # mpl.draw()
            fig1.savefig(str(config.batch_size) + '_' + str(config.num_steps) + 'run1_nonPercentChange.png')

            performance = self.calcPerformance(self.test_Y, final_prediction)
            print("Directional accuracy of predictions")
            print("NumCorrect: ", performance[0])
            print("Total: ", performance[1])
            print("Percentage: ", float(performance[0])/float(performance[1]))

            self.performance = float(performance[0])/float(performance[1])

            #  print("Backtest starting with $10000", self.backTest(self.testPrices, _pred))

            """graph_saver_dir = os.path.join(config.MODEL_DIR, graph_name)
            if not os.path.exists(graph_saver_dir):
                os.mkdir(graph_saver_dir)

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(
                graph_saver_dir, "stock_rnn_model_%s.ckpt" % graph_name), global_step=epoch_step)

        with open("final_predictions.{}.json".format(graph_name), 'w') as fout:
            fout.write(json.dumps(final_prediction.tolist()))"""

    def calcPerformance(self, actual, predicted, train=False):
        # print(output)
        # Outdated code for when inputs were percentage change
        """
        assert(predicted.shape == actual.shape)
        totalPreds = len(predicted)
        numCorrect = 0
        for y, pred in zip(actual,predicted):
            if(np.sign(y) == np.sign(pred)):
                numCorrect += 1
        return numCorrect,totalPreds
        """

        assert (predicted.shape == actual.shape)
        #Denormalization
        #actual = (actual * self.stdev[0]) + self.mean[0]
        #predicted = (predicted * self.stdev[0]) + self.mean[0]
        # timesteps = np.arange(actual.shape[0])
        # mpl.plot(timesteps, np.stack(actual), label='data')
        # mpl.plot(timesteps, np.stack(predicted), label='predict')
        # mpl.xlabel('Time')
        # mpl.ylabel('Normalized price change')
        # mpl.legend()
        # mpl.show()
        # mpl.draw()
        totalPreds = len(predicted)
        numCorrect = 0

        profit = 0

        """
        Denormalize
        """
        # print("std dev of predictions, pre-denormal", np.std(predicted))
        # print("std dev of actual, pre-denormal", np.std(actual))

        zeros = np.zeros((predicted.shape[0], config.input_size))
        zeros[:,0] = np.reshape(predicted, (-1))
        predicted = zeros

        newZeros = np.zeros((actual.shape[0], config.input_size))
        newZeros[:,0] = np.reshape(actual, (-1))
        actual = newZeros



        predicted = self.scaler.inverse_transform(predicted)

        actual = self.scaler.inverse_transform(actual)

        predicted = predicted[:,0]
        actual = actual[:,0]


        # actual = actual + self.mean[0]

        # print(predicted[:5])
        # print(actual[:5])

        last = actual[0]
        for i in range(len(actual)):
            # if (np.sign(predicted[i] - last) == np.sign(actual[i] - last)):
            #     numCorrect += 1
            if(np.sign(predicted[i] - np.float32(1)) == np.sign(actual[i] - np.float32(1))):
                numCorrect += 1
            elif(np.sign(predicted[i] - np.float(1)) == 0):
                numCorrect += 0.5
            # else:
            #     print('WRONG', predicted[i], actual[i])
            # profit += np.sign(predicted[i] - last)*(actual[i] - last)
            # last = actual[i]
        # print("amount gained/lost", profit)
        return numCorrect, totalPreds

    def statsFeatures(self, input, config):
        slow_williamsR = []
        RoC = []
        momentum = []
        fast_williamsR = []
        RSI = []
        TEMA = []

        upChangeAvg = 0
        downChangeAvg = 0

        alpha = 0.2
        lastPrice = input[0,0]
        priceAvg = input[0,0]

        ema_ema = input[0,0]
        ema_ema_ema = input[0,0]
        for i in range(input.shape[0]):

            """
            SLOW WILLIAMS
            """
            if(i == 0):
                highest = input[0,0]
                lowest = input[0,0]
                slow_williamsR.append(np.float32(-0.5))
            elif(i < config.williams_period):
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
            elif (i < (config.williams_period/2)):
                highest = np.amax(input[:i+1, 0])
                lowest = np.amin(input[:i+1, 0])
                fast_williamsR.append(-1. * (highest - input[i, 0]) / (highest - lowest))
            else:
                highest = np.amax(input[i - int(config.williams_period/2):i+1, 0])
                lowest = np.amin(input[i - int(config.williams_period/2):i+1, 0])
                fast_williamsR.append(-1. * (highest - input[i, 0]) / (highest - lowest))

            if(input[i,0] > lastPrice):
                upChangeAvg = (alpha*(input[i,0]-lastPrice)) + (1-alpha)*upChangeAvg
                downChangeAvg = (1-alpha)*downChangeAvg
            else:
                downChangeAvg = (alpha * (lastPrice - input[i, 0])) + (1 - alpha) * downChangeAvg
                upChangeAvg = (1 - alpha) * upChangeAvg
            if(downChangeAvg == 0):
                RS = 2
            else:
                RS = upChangeAvg/downChangeAvg
            RSI_calc = 100 - 100./(1+RS)
            RSI.append(RSI_calc)

            priceAvg = (alpha*input[i,0]) + (1-alpha)*priceAvg
            ema_ema = (alpha*priceAvg) + (1-alpha)*ema_ema
            ema_ema_ema = (alpha*ema_ema) + (1-alpha)*ema_ema_ema

            TEMA_calc = 3*priceAvg - 3*ema_ema + ema_ema_ema
            TEMA.append(TEMA_calc)

            lastPrice = input[i,0]

        #Currently hard-coded in input
        allFeatures = np.array(list(zip(input[:,0], input[:,1], slow_williamsR, RoC, momentum, fast_williamsR, RSI, TEMA)))

        #allFeatures = np.stack((slow_williamsR, RoC), -1)
        #print(allFeatures.shape)
        #allFeatures = np.concatenate((allFeatures, momentum), -1)
        #print(allFeatures.shape)
        #allFeatures = np.concatenate((allFeatures, fast_williamsR), -1)
        #print(allFeatures.shape)
        #allFeatures = np.concatenate((allFeatures, RSI), -1)
        #print(allFeatures.shape)
        #allFeatures = np.concatenate((allFeatures, TEMA), -1)
        #print(allFeatures.shape)
        #print(allFeatures[0])

        return allFeatures

    def backTest(self, stockPrices, normPredicted):
        normPredicted = np.squeeze(normPredicted)
        # print(stockPrices.shape)
        # print(normPredicted.shape)
        assert(stockPrices.shape == normPredicted.shape)

        cash = 10000
        numStocks = 0

        for i in range(stockPrices.shape[0]):
            if(normPredicted[i] > 0):
                cash -= stockPrices[i]*50*normPredicted[i]
                numStocks += 50*normPredicted[i]
            elif(normPredicted[i] < 0):
                if not (numStocks == 0):
                    cash += stockPrices[i]*50*np.abs(normPredicted[i])
                    numStocks -= 50*np.abs(normPredicted[i])
        return cash + (numStocks*stockPrices[-1])

class config:
    keep_prob = 0.85
    input_size = 8
    output_size = 1
    batch_size = 1
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    lstm_size = 185
    num_layers = 1
    num_steps = 1
    test_train_split = 0.8
    MODEL_DIR = os.getcwd() + "\\models"
    init_epoch = 5
    #TEMP
    max_epoch = 2000
    train_validation_split = 0.8 # Set to 0 if not running validation
    early_stopping_continue = 100

    #TESTING
    sin_data = False

    #FEATURES
    williams_period = 10

    stock = "MSFT"

def sinData(numSteps):
    time = np.arange(numSteps)
    time = time/10
    #print(time)
    list = (np.sin(time) + np.random.normal(0,0.1,size=(len(time))))+np.log(time+2)
    someNum = np.zeros(shape=list.shape) + np.random.normal(0,0.0001,size=(len(time)))

    stacked = np.stack((time, list, someNum), axis=-1)
    mpl.plot(time, list)
    mpl.title("Noisy Sine Data")
    # mpl.show()
    # mpl.draw()
    return pd.DataFrame(stacked)



if __name__ == "__main__":
    tf.reset_default_graph()
    """
    Testing purposes
    
    """



    """
    Hyper parameter tuning
    """

    errorDict = {}
    #Random search
    for n in range(50):
        n = LSTM_System()
        n.config.num_layers = 2 #np.random.choice(4, p=[0.5, 0.25, 0.125, 0.125]) + 1
        n.config.batch_size = 1 #np.random.randint(1,300/n.config.num_layers)
        #n.config.num_steps = np.random.randint(1,n.config.batch_size+1)
        n.config.keep_prob = np.random.uniform(0.3,1)
        n.config.lstm_size = np.random.randint(1, 512/n.config.num_layers)
        n.config.init_learning_rate = np.random.uniform(0.0005,0.01)
        n.config.max_epoch = int(float(1000)/n.config.num_layers)
        n.config.early_stopping_continue = int(float(100)/n.config.num_layers)
        #Currently just testing microsoft
        #n.config.stock = n._UsedStocks[np.random.randint(0,len(n._UsedStocks))]


        print("\n\n", n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers,
              n.config.init_learning_rate)
        lstm_graph = n.rnn(config=n.config)
        n.train_lstm(lstm_graph, config=n.config)
        errorDict[n.performance] = str(n.config.batch_size)+","+str(n.config.num_steps)+","+str(n.config.keep_prob)+","+\
                                  str(n.config.lstm_size)+","+str(n.config.num_layers)+","+str(n.config.init_learning_rate)
        print(n.final_loss)
        print("\nDONE\n", n.config.batch_size, n.config.num_steps, n.config.keep_prob, n.config.lstm_size, n.config.num_layers,
              n.config.init_learning_rate)
    keyList = (sorted(list(errorDict.keys()), reverse=True))
    for i in range(8):
        print(errorDict[keyList[i]], "Directional Accuracy: ", keyList[i])


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
