import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import paths_constants

class EvaluateModel:
    def __init__(self, dataset_file, execution_type):
        valid = {'strongly', 'weakly'}
        if execution_type not in valid:
            raise ValueError("EvaluateModel: execution_type must be one of %r." % valid)
        
        self.dataset_file = dataset_file
        self.execution_type = execution_type
        self.neural_network_model_path = ''
        self.dataframe = pd.DataFrame({})
        self.metrics_axis = dict()
    
    def load_execution(self, epoch_runs, train_folds, validation_folds, test_folds):
        neural_network_model_file = paths_constants.neural_network_model_file(
            self.dataset_file.stem,
            self.execution_type,
            epoch_runs,
            train_folds,
            validation_folds,
            test_folds
        )
        self.neural_network_model_path = neural_network_model_file.parent
        self.dataframe = pd.read_csv(neural_network_model_file, sep=' ', names=['output','is_bot'])
        self.dataframe = self.dataframe.sort_values(by='output')
        self.metrics_axis = self.__calculate_matric_axis()

    def metrics_for_threshold(self, threshold):
        classified_bot  = self.dataframe[self.dataframe['output'] >= threshold]['is_bot']
        classified_user = self.dataframe[self.dataframe['output'] < threshold]['is_bot']

        true_positive  = len(np.where(classified_bot == 1)[0]) # bots classified as bots
        false_positive = len(np.where(classified_bot == 0)[0]) # users classified as bots

        false_negative = len(np.where(classified_user == 1)[0]) # bots classified as users
        true_negative  = len(np.where(classified_user == 0)[0]) # users classified as users

        recall_bot = true_positive / (true_positive + false_negative)
        precision  = true_positive / (true_positive + false_positive)
        f1_score   = 2 * (precision * recall_bot) / (precision + recall_bot)

        recall_not_bot = true_negative / (true_negative + false_positive)

        return {
            'recall_bot': recall_bot,
            'recall_not_bot': recall_not_bot,
            'precision': precision,
            'f1_score': f1_score
        }
    
    def __calculate_matric_axis(self):
        x_threshold = []
        y_recall_bot = []
        y_recall_not_bot = []
        y_precision = []
        y_f1_score = []

        granularity = 0.0005
        decimal_places = 4
        for threshold in np.arange(0, 1 + granularity, granularity):
            threshold = round(threshold, decimal_places)

            metrics = self.metrics_for_threshold(threshold)

            x_threshold.append(threshold)
            y_recall_bot.append(metrics['recall_bot'])
            y_recall_not_bot.append(metrics['recall_not_bot'])
            y_precision.append(metrics['precision'])
            y_f1_score.append(metrics['f1_score'])
        
        return {
            'x_threshold': x_threshold,
            'y_recall_bot': y_recall_bot,
            'y_recall_not_bot': y_recall_not_bot,
            'y_precision': y_precision,
            'y_f1_score': y_f1_score
        }

    def __roc_curve_auc(self):
        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(self.dataframe['is_bot'], self.dataframe['output'], pos_label=1)
        auc = metrics.auc(false_positive_rates, true_positive_rates)
        return false_positive_rates, true_positive_rates, auc
    
    def calculate_auc(self):
        _, _, auc = self.__roc_curve_auc()
        return auc

    def plot_roc_curve(self, save=False):
        false_positive_rates, true_positive_rates, auc = self.__roc_curve_auc()

        fig = plt.figure(figsize=(12,8))
        plt.tight_layout()

        plt.xticks(np.linspace(0, 1, num=11))
        plt.yticks(np.linspace(0, 1, num=11))

        plt.plot(false_positive_rates, true_positive_rates, label = 'ROC (AUC = ' + str(auc) + ')')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend()

        if save:
            self.__save_plot(plt, 'roc_curve')
        
        plt.show()
        plt.clf()
    
    def plot_evaluation_metrics(self, save=False, show_bot_recall=True, show_not_bot_recall=True, show_precision=True, show_f1_score=True, vertical_line=None):
        fig = plt.figure(figsize=(12,8))
        plt.tight_layout()
        
        plt.xticks(np.linspace(0, 1, num=11))
        plt.yticks(np.linspace(0, 1, num=11))
        
        if show_bot_recall:
            plt.plot(self.metrics_axis['x_threshold'], self.metrics_axis['y_recall_bot'], c='orange', label='Bot recall')
        if show_not_bot_recall:
            plt.plot(self.metrics_axis['x_threshold'], self.metrics_axis['y_recall_not_bot'], c='gray', label='Not bot recall')
        if show_precision:
            plt.plot(self.metrics_axis['x_threshold'], self.metrics_axis['y_precision'], c='blue', label='Precision')
        if show_f1_score:
            plt.plot(self.metrics_axis['x_threshold'], self.metrics_axis['y_f1_score'], c='red', label='F1 score')
        
        if vertical_line:
            plt.axvline(vertical_line, linestyle=':')
        
        plt.xlabel('Acceptance Threshold for Bot')
        plt.legend()
        
        if save:
            filename = 'metrics'
            if show_bot_recall:
                filename += '-bot_recall'
            if show_not_bot_recall:
                filename += '-not_bot_recall'
            if show_precision:
                filename += '-bot_precision'
            if show_f1_score:
                filename += '-f1_score'
            if vertical_line:
                filename += '-line_' + str(vertical_line)
            self.__save_plot(plt, filename)
        
        plt.show()
        plt.clf()
    
    def __save_plot(self, plot, chart_type):
        filename = chart_type + '.png'
        plot.savefig(self.neural_network_model_path / filename)


class EvaluateModelWeaklyComponent(EvaluateModel):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'weakly')


class EvaluateModelStronglyComponent(EvaluateModel):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'strongly')
