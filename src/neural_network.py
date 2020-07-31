import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tflearn.activations import softmax
from tflearn.layers.core import fully_connected
from sklearn.model_selection import StratifiedKFold

import paths_constants
from dataframe_reader import DataframeReader

class NeuralNetwork(DataframeReader):
    def __init__(self, dataset_file, execution_type):
        self.parameters_resume = ''
        self.model_Y_test = np.array([])
        self.model_Y_test_output = np.array([])
        super().__init__(dataset_file, execution_type)

    def plot_correlation_matrix(self, save=False):
        plt.figure(figsize=(16,9))

        corr_matrix = self.dataframe.corr()
        sns.heatmap(
            corr_matrix,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            annot=True,
            cbar_kws={'ticks': np.linspace(-1, 1, num=11)}
        )

        plt.xticks(rotation=70)
        plt.tight_layout()

        if save:
            self.__save_plot(plt, 'corr_matrix')

        plt.show()
        plt.clf()


    def __save_plot(self, plot, chart_type):
        neural_network_path = paths_constants.neural_network(self.dataset_file.stem, self.execution_type)
        neural_network_path.mkdir(parents=True, exist_ok=True)
        filename = chart_type + '.png'
        plot.savefig(neural_network_path / filename)


    def normalize_dataframe(self):
        columns_to_normalize = list(self.dataframe.columns)
        columns_to_normalize.remove('is_bot')
        self.dataframe[columns_to_normalize] = (
            (self.dataframe[columns_to_normalize] - self.dataframe[columns_to_normalize].min()) /
            (self.dataframe[columns_to_normalize].max() - self.dataframe[columns_to_normalize].min())
        )
    

    def __neural_network_model(self, user_class_weight):
        print('Building convolutional network...')

        X = tf.placeholder(tf.float32, [None, 12], name="X")
        Y = tf.placeholder(tf.float32, [None, 2], name="Y")

        MLP1 = fully_connected(X, 64, activation=tf.nn.elu, regularizer="L2", name="MLP1")

        MLP2 = fully_connected(MLP1, 2, activation='linear', regularizer="L2", name="MLP2")

        output = softmax(MLP2)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32),
            name='acc'
        )

        class_weights = tf.constant([[user_class_weight, 1.0]])
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * Y, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=MLP2)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)
    
        return {
            'X': X,
            'Y': Y,
            'output': output,
            'accuracy': accuracy,
            'loss': loss
        }
    

    def __train_model(self, sess, model, optimizer, X, Y, batch_size=100):
        total_batch = int(len(X) / batch_size)
        train_order = np.random.permutation(X.shape[0])
        for i in train_order:
            ini = i * batch_size
            if ini >= len(X):
                continue
            fin = (i + 1) * batch_size
            batch_x = X[ini:fin]
            batch_y = Y[ini:fin]
            sess.run(optimizer, feed_dict={model['X']: batch_x, model['Y']: batch_y})
            ini = fin
    

    def __calculate_recall(self, labels, output):
        b_c1 = output[:,0]
        b_c2 = output[:,1]

        rec1 = len(np.where((labels[:,0] == b_c1) & (labels[:,0] == 1))[0])
        totalc1 = len(np.where((labels[:,0] == 1))[0])

        rec2 = len(np.where((labels[:,1] == b_c2) & (labels[:,1] == 1))[0])
        totalc2 = len(np.where((labels[:,1] == 1))[0])

        recall_1 = rec1 / totalc1 if totalc1 > 0 else 0

        recall_2 = rec2 / totalc2 if totalc2 > 0 else 0

        return np.array([recall_1, recall_2])


    def __evaluate_model(self, sess, model, X, Y, batch_size=100):  
        total_batch = int(len(X)/batch_size)
        ini = 0
        cost = 0
        acc = 0
        total_batches = 0
        recall = np.array([0.,0.])
        
        for i in range(total_batch+1):
            if ini>=len(X):
                continue

            fin = (i + 1) * batch_size
            batch_x = X[ini:fin]
            batch_y = Y[ini:fin]
            batch_cost, batch_acc, batch_output = sess.run(
                [model['loss'], model['accuracy'], model['output']],
                feed_dict={model['X']: batch_x, model['Y']: batch_y}
            )

            batch_recall = self.__calculate_recall(batch_y, batch_output)

            try:
                output = np.concatenate((output, batch_output), axis=0)
            except:
                output = batch_output

            recall += batch_recall
            cost += batch_cost
            acc += batch_acc

            total_batches += 1
            ini = fin

        cost = cost / total_batches
        acc = acc / total_batches   
        recall = recall / total_batches

        return cost, acc, recall, output
    

    def __prepare_data(self, train, val, test):
        print('Preparing data from the dataframe...')

        total_folds = train + val + test

        columns = list(self.dataframe.columns)
        columns.remove('is_bot')

        X = self.dataframe[columns].values

        Y0 = self.dataframe['is_bot'].values.reshape(-1, 1)
        Y1 = np.abs(Y0 - 1).reshape(-1, 1)
        Y = np.concatenate((Y0, Y1), axis=1)

        train_val_split = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=4567)

        folds = []
        for dev_index, test_index in train_val_split.split(X, Y0):
            folds.append(test_index)
        
        X_train = X[np.concatenate([folds[i] for i in range(train)], axis=0)]
        X_val   = X[np.concatenate([folds[i] for i in range(train, train+val)], axis=0)]
        X_test  = X[np.concatenate([folds[i] for i in range(train+val, total_folds)], axis=0)]

        Y_train = Y[np.concatenate([folds[i] for i in range(train)], axis=0)]
        Y_val   = Y[np.concatenate([folds[i] for i in range(train, train+val)], axis=0)]
        Y_test  = Y[np.concatenate([folds[i] for i in range(train+val, total_folds)], axis=0)]

        y0_uniques, network_users_bots_distribution = np.unique(Y0, return_counts=True)
        network_total_users, network_total_bots = network_users_bots_distribution
        network_user_class_weight = network_total_users / network_total_bots

        print('Network has:', network_total_users, 'users and', network_total_bots, 'bots.')

        return X_train, X_val, X_test, Y_train, Y_val, Y_test, network_user_class_weight
    

    def execute_neural_network(self, epoch_runs, train_folds, validation_folds, test_folds, show_progress=False):
        fixed_seed = 123

        tf.reset_default_graph()
        
        np.random.seed(fixed_seed)
        tf.set_random_seed(fixed_seed)

        X_train, X_val, X_test, Y_train, Y_val, Y_test, network_user_class_weight = self.__prepare_data(train_folds, validation_folds, test_folds)

        model = self.__neural_network_model(network_user_class_weight)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(model['loss'])
        
        # training
        init = tf.global_variables_initializer()
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        # launch the graph
        saver = tf.compat.v1.train.Saver()
        model_results = []
        
        with sess.as_default():
            sess.run(init) 

            print("Starting epoch executions...")

            if show_progress:
                acc_ant = 0
                cost_ant = 1000
                print('\nShowing progress of epochs with validation_cost improvements...')
                print('Epoch \t|\t train_accuracy \t|\t validation_accuracy \t|\t test_accuracy')

            for epoch in range(epoch_runs):
                self.__train_model(sess, model, optimizer, X_train, Y_train, 400)

                train_cost, train_acc, train_rec, train_output = self.__evaluate_model(sess, model, X_train, Y_train, 400)
                val_cost, val_acc, val_rec, val_output         = self.__evaluate_model(sess, model, X_val, Y_val, 400)
                test_cost, test_acc, test_rec, test_output     = self.__evaluate_model(sess, model, X_test, Y_test, 400)
                
                if show_progress:
                    is_better_cost = False
                    is_better_acc = False

                    if val_cost <= cost_ant:
                        cost_ant = val_cost
                        if show_progress:
                            print(epoch + 1, '\t|\t', train_acc, '\t|\t', val_acc, '\t|\t', test_acc)
                        is_better_cost = True
                        
                    if val_acc >= acc_ant:
                        acc_ant = val_acc
                        is_better_acc = True
    
                    model_results.append([
                        is_better_cost,
                        is_better_acc,
                        train_acc,
                        train_cost,
                        val_acc,
                        val_cost,
                        test_acc,
                        test_cost
                    ])
  
        print('Finishing neural network execution...')
        if show_progress:
            df_results = pd.DataFrame(
                model_results,
                columns=[
                    'is_better_cost',
                    'is_better_acc',
                    'train_acc',
                    'train_cost',
                    'val_acc',
                    'val_cost',
                    'test_acc',
                    'test_cost'
                ]
            )
            df_results.index = np.arange(1, len(df_results) + 1)

            print('\n\nLast epoch with validation_accuracy improvements...')
            print(df_results[(df_results['is_better_acc']==True)][-1:].to_string())
            print('\nLast epoch with validation_cost improvements...')
            print(df_results[(df_results['is_better_cost']==True)][-1:].to_string())

        self.parameters_resume = str(epoch_runs) + '-' + str(train_folds) + '-' + str(validation_folds) + '-' + str(test_folds)
        self.model_Y_test = Y_test[:,0]
        self.model_Y_test_output = test_output[:,0]
    

    def save_results_from_model(self):
        if self.model_Y_test.size == 0 or self.model_Y_test_output.size == 0:
           print('Run execute_neural_network first. Current model_Y_test or model_Y_test_output are empty.')
           return 

        neural_network_path = paths_constants.neural_network(self.dataset_file.stem, self.execution_type)
        neural_network_path.mkdir(parents=True, exist_ok=True)
        filename = self.parameters_resume + '_neural_network_results_(y_output-y_test)'

        with open(neural_network_path / filename, 'w') as file:
            for index, test_output in enumerate(self.model_Y_test_output):
                file.write(str(test_output) + ' ' + str(self.model_Y_test[index]) + '\n')


class NeuralNetworkWeaklyComponent(NeuralNetwork):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'weakly')


class NeuralNetworkStronglyComponent(NeuralNetwork):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'strongly')
