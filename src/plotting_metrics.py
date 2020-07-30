import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths_constants
from utils import *
from dataframe_reader import DataframeReader

class PlottingMetrics(DataframeReader):
    
    def print_single_value_metric(self, metric_name):
        subgraph_metrics_path = paths_constants.metrics_subgraph(self.dataset_file.stem, self.execution_type)
        print('Metric: ', end = '')
        with open(subgraph_metrics_path / (metric_name)) as file:
            for line in file:
                print(line.strip())
        print('\n')

    def print_metric_summary(self, metric_name):
        subgraph_metrics_path = paths_constants.metrics_subgraph(self.dataset_file.stem, self.execution_type)
        print('Summary: ', end = '')
        with open(subgraph_metrics_path / (metric_name + '_summary')) as file:
            for line in file:
                print(line.strip())
        print('\n')
    
    def __get_values_for_users_and_bots(self, metric):
        info_user = [value for username, value in self.dataframe[metric].iteritems() if self.dataframe['is_bot'][username] == 0]
        info_bot  = [value for username, value in self.dataframe[metric].iteritems() if self.dataframe['is_bot'][username] == 1]
        return info_user, info_bot

    def plot_ccdf(self, metric_1, metric_2=None, save=False):
        plt.figure(figsize=(8,6))
        plt.grid(color='gray', linestyle=':')
        plt.ylabel('Fraction of Users ≥ k')

        self.print_metric_summary(metric_1)
        info_user_1, info_bot_1 = self.__get_values_for_users_and_bots(metric_1)
        x_user_1, ccdf_user_1 = ccdf(info_user_1)
        x_bot_1,  ccdf_bot_1  = ccdf(info_bot_1)
        
        xlabel_1 = metric_1.replace('_', ' ').title()
        plt.xlabel(xlabel_1)
        
        plt.loglog(x_user_1, ccdf_user_1, '.', c = 'blue', marker='o', label='User (' + xlabel_1 + ')')
        plt.loglog(x_bot_1, ccdf_bot_1, '.', c = 'red', marker='o', label='Bot (' + xlabel_1 + ')')
        
        if metric_2:
            self.print_metric_summary(metric_2)
            info_user_2, info_bot_2 = self.__get_values_for_users_and_bots(metric_2)
            x_user_2, ccdf_user_2 = ccdf(info_user_2)
            x_bot_2,  ccdf_bot_2  = ccdf(info_bot_2)

            xlabel_2 = metric_2.replace('_', ' ').title()
            plt.xlabel(xlabel_1 + ' (Circle), ' + xlabel_2 + ' (Triangle)')

            plt.loglog(x_user_2, ccdf_user_2, '.', c = 'darkblue', marker='^', label='User (' + xlabel_2 + ')')
            plt.loglog(x_bot_2, ccdf_bot_2, '.', c = 'darkred', marker='^', label='Bot (' + xlabel_2 + ')')
        
        plt.tick_params(direction='in')
        plt.legend()
        
        if save:
            self.__save_plot(plt, 'ccdf', metric_1, metric_2)
        
        plt.show()
        plt.clf()


    def plot_scatter(self, metric_x, metric_y, save=False):
        self.print_metric_summary(metric_x)
        self.print_metric_summary(metric_y)
        
        x_user, y_user, x_bot, y_bot = ([] for _ in range(4)) # empty lists
        
        for username in self.dataframe.index:
            if self.dataframe['is_bot'][username]:
                x_bot.append(self.dataframe[metric_x][username])
                y_bot.append(self.dataframe[metric_y][username])
            else:
                x_user.append(self.dataframe[metric_x][username])
                y_user.append(self.dataframe[metric_y][username])
        
        plt.figure(figsize=(8,6))
        plt.grid(color='gray', linestyle=':')
    
        xlabel = metric_x.replace('_', ' ').title()
        ylabel = metric_y.replace('_', ' ').title()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.loglog(x_user, y_user, '.', c = 'blue', marker='x', label='User')
        plt.loglog(x_bot, y_bot, '.', c = 'red', marker='x', label='Bot')
        
        plt.tick_params(direction='in')
        
        plt.legend(loc='upper right')
        
        if save:
            self.__save_plot(plt, 'scatter', metric_x, metric_y)
        
        plt.show()
        plt.clf()

    
    def __save_plot(self, plot, chart_type, metric_1, metric_2):
        charts_subgraph_path = paths_constants.charts_subgraph(self.dataset_file.stem, self.execution_type)
        charts_subgraph_path.mkdir(parents=True, exist_ok=True)
        filename = (chart_type + '_' + metric_1 + '_X_' + metric_2 + '.png') if metric_2 else (chart_type + '_' + metric_1 + '.png')
        plot.savefig(charts_subgraph_path / filename)


class PlottingWeaklyMetrics(PlottingMetrics):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'weakly')


class PlottingStronglyMetrics(PlottingMetrics):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'strongly')