import gym
from gym import spaces
import numpy as np
from numpy import linalg
from numpy.random import randn
from random import randint
# import scipy.stats
import os, math, random, itertools, csv, pickle, inspect, torch
from itertools import combinations, permutations, product
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import multiprocessing
from tqdm import tqdm
import concurrent.futures
import datetime
from datetime import date
import scipy
from sklearn import metrics
import pandas as pd
# from keras import models
from keras.models import Sequential, model_from_json
from keras.layers import LSTM
from keras.layers import Bidirectional
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from joblib import dump, load
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import math
from scipy.spatial import distance
# import SCandPowerAll
from numpy import random as rnd
import networkx as nx
from pulp import *
# from tensorflow.keras import layers
# from keras.saving import save as save_module
import peewee
import sys
import scipy.io
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import statsmodels.api as sm

# import BiLSTM
# from keras.layers.convolutional_recurrent import ConvLSTM2D

# @@@@@@@@@@@@@@@
import keras
# from keras import models
import joblib
import random
# from keras.saving import save as save_module
# import matlab.engine
import multiprocessing as mp
from keras.layers import Dense, Activation, BatchNormalization, Input, LSTM, Bidirectional, TimeDistributed, Flatten
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator
from keras import Sequential
from keras.callbacks import EarlyStopping
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.framework import ops
from matplotlib import ticker
# import cv2
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'figure.figsize': [12, 8],  # width, height in inches
    'axes.grid': True,
    'lines.linewidth': 3.5,
    'lines.markersize': 13,
    'figure.subplot.wspace': 0.5,
})

#####################################
# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



def load_model_predict(data_id, key_before_lab):
    pred_gamma = {}
    g_lstm_b_u = []
    g_lstm_f_u = []
    g_Bilstm_u = []

    # Get last N elements from given list #reshape the nonuniform 2D array for all UEs per genre g
    # g_k = [i[-len(min(g_k, key=len)):] for i in  g_k]  # columns are time series to the right most recent, rows are different UEs
    Xo = [i[-window_size_b:] for i in X if len(i) >= window_size_b]
    X = [i[-window_size:] for i in X if len(i) >= window_size]
    # X = np.array(X).reshape((np.array(X).shape[0], 1, np.array(X).shape[1]))
    Y = np.mean(X)

    # D:\CQUPT\ModifiedResearchCode\CelFreeCLCA_RL-master\TrainedPopularModels
    file_directory = "D:\CQUPT\CQUPT_FinalCode\TrainedBiModels/LSTM_backward_" + str \
        (g) + ".h5"
    Trained_model = keras.models.load_model(file_directory)
    X = np.array(X).reshape((np.array(X).shape[0], 1, np.array(X).shape[1]))
    Xo = np.array(Xo).reshape((np.array(Xo).shape[0], 1, np.array(Xo).shape[1]))
    y_predicted = Trained_model.predict(
        X)  # predicted the next requet probability of all files  result = loaded_model.score
    g_pred_y_lstm_b_u = np.hstack(y_predicted.tolist())

    # D:\CQUPT\ModifiedResearchCode\CelFreeCLCA_RL - master\TrainedPopularModels
    file_directory = "D:\CQUPT\CQUPT_FinalCode\TrainedPopularModels/LSTM_Forward_" + str \
        (g) + ".h5"
    Trained_model = keras.models.load_model(file_directory)
    y_predicted = Trained_model.predict(X)  # predicted the next requet probability of all files
    # y_predicted_class_lstm_back = Trained_model_lstm_back.predict_classes(X)
    g_pred_y_lstm_f_u = np.hstack(y_predicted.tolist())

    # X = np.array(X).reshape((np.array(Xo).shape[0], 1, np.array(Xo).shape[1]))
    # D:\CQUPT\ModifiedResearchCode\CelFreeCLCA_RL - master\TrainedPopularModels

    file_directory = "J:\CQUPT\CQUPT_FinalCode\TrainedModels/BiLSTM_" + str(g) + ".h5"
    # file_directory = "D:\CQUPT\CQUPT_FinalCode\TrainedBiModels\BiLSTM_" + str(g) + ".h5"

    Trained_model = keras.models.load_model(file_directory)
    y_predicted = Trained_model.predict(X)  # predicted the next requet probability of all files
    # y_predicted_class_lstm_back = Trained_model_lstm_back.predict_classes(X)
    g_pred_y_Bilstm_u = np.hstack(y_predicted.tolist())

    g_lstm_b_u.append(g_pred_y_lstm_b_u)
    g_lstm_f_u.append(g_pred_y_lstm_f_u)
    g_Bilstm_u.append(g_pred_y_Bilstm_u)

    pred_gamma["lstm_back"] = g_lstm_b_u
    pred_gamma["lstm_Forward"] = g_lstm_f_u
    pred_gamma["Bilstm"] = g_Bilstm_u

    return pred_gamma


# IMPORTANT NOTE
# Binary Keras LSTM model does not output binary predictions
# function to create input sequences and output labels
"""def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)"""


def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(0, len(data) - time_steps, time_steps):
        X.append(data[i:i + time_steps])
        y_i = np.mean(data[i + time_steps:i + time_steps * 2], axis=0)
        y_i /= np.sum(y_i)  # Normalize the sum of y_i to 1
        y.append(y_i)
    return np.array(X), np.array(y)


def prepare_data_model(learning_rate):
    """##################################################################################
    ratings = pd.read_csv('ml-latest/ratings.csv').sample(frac=0.05)
    movies = pd.read_csv('ml-latest/movies.csv').sample(frac=1)

    # clean Data
    data = pd.merge(ratings, movies, how='inner')
    data = data.drop(['title'], axis=1)
    data = data[~data.genres.str.contains("no genres listed", na=False)]
    # unique, data_c = np.unique(data['genres'], return_counts=True)
    # dict(zip(unique, data_c))
    data = data.drop(['rating'], axis=1)

    data['timestamp'] = data['timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d '))  # %H:%M:%S
    data['timestamp'].astype('datetime64[ns]')
    dummies = data['genres'].str.get_dummies(sep='|')
    data = pd.concat([data, dummies], axis=1).drop(columns=['genres'])
    data = data.dropna()

    data_i = np.array(data)
    data_i = pd.DataFrame(data_i)
    #data_i.to_csv("F:/CQUPT/NEW_REVISED_CODE/ml-latest/data100ML.csv")
    data_i.to_csv("F:/CQUPT/NEW_REVISED_CODE/ml-latest/data27ML.csv")"""

    # read
    """data = pd.read_csv(
        "data27ML.csv",
        index_col=0, delimiter=',')
    data = pd.DataFrame(data)"""

    ##################################################################################
    agg_funcs = {'Action': 'sum', 'Adventure': 'sum', 'Animation': 'sum', 'Children': 'sum', 'Comedy': 'sum',
                 'Crime': 'sum', 'Documentary': 'sum', 'Drama': 'sum', 'Fantasy': 'sum', 'Film-Noir': 'sum',
                 'Horror': 'sum', 'IMAX': 'sum', 'Musical': 'sum', 'Mystery': 'sum', 'Romance': 'sum',
                 'Sci-Fi': 'sum', 'Thriller': 'sum', 'War': 'sum', 'Western': 'sum'}

    key_before_label = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                        'War', 'Western']

    dict_movie_count = {}
    for genre in key_before_label:
        count = 0
        for d_m in np.unique(data['movieId']):
            if np.sum(data[data['movieId'] == d_m][genre]) != 0:
                count += 1
        dict_movie_count[str(genre)] = count
    data_plot = data.groupby(['userId', 'timestamp'], as_index=False).agg(agg_funcs)

    dict_g = {}
    for g, genres in enumerate(key_before_label):
        dict_g[str(genres)] = np.sum(data_plot[key_before_label[g]])

    key_before_labe_=[]
    for d_mc,k in zip(dict_movie_count.values(), key_before_label):
        if d_mc<800:
            data_plot = data_plot.drop([str(k)], axis=1)
        else:
            key_before_labe_.append(k)

    dict_g_popular = {}
    for g, genres in enumerate(key_before_labe_):
        dict_g_popular[str(genres)] = np.sum(data_plot[key_before_labe_[g]])


    fig0 = plt.figure(0)
    ax1 = fig0.add_subplot(1, 2, 1)
    ax2 = fig0.add_subplot(1, 2, 2)

    # print(dict_genere)
    ax1.bar(dict_movie_count.keys(), dict_movie_count.values(), color='skyblue')
    ax1.set_xticks(range(len(dict_movie_count.keys())), dict_movie_count.keys(), rotation=90)
    ax1.set_ylabel("Number of Movies per Genre")

    # print(dict_genere)
    ax2.bar(dict_g.keys(), dict_g.values(), color='skyblue')
    ax2.set_xticks(range(len(dict_g.keys())), dict_g.keys(), rotation=90)
    ax2.set_ylabel("View Counts of Movies per Genre")

    figu = plt.figure(1)
    ax3 = figu.add_subplot(1, 1, 1)
    ax3.bar(dict_g_popular.keys(), dict_g_popular.values(), color='skyblue')
    ax3.set_xticks(range(len(dict_g_popular.keys())), dict_g_popular.keys(), rotation=90)
    ax3.set_ylabel("View Counts of Movies per Genre")

    fig0.tight_layout()
    figu.tight_layout()
    plt.show()

    window_size = 5
    # data = data.rename(columns={data.keys()[10]: 'Children'})
    ##################################################################################
    # convert timestamp to datetime object
    data['timestamp'] = pd.to_datetime(data['timestamp'])


    # Create a dictionary of dataframes, with one dataframe for each user
    data_id = {}
    for id in np.unique(data['userId']):
        user_data = data.loc[data['userId'] == id]
        if len(user_data) >= window_size:
            user_data = pd.DataFrame(user_data.drop(['userId', 'movieId'], axis=1))

            user_data.set_index('timestamp', inplace=True)  # df.sort_values(by='timestamp', ascending=False)
            user_data_yearly = user_data.resample('Y').sum()

            user_data_yearly = user_data_yearly.loc[:, 'Action':'Western']
            # user_data_yearly =  user_data_yearly.loc[(user_data_yearly != 0).any(axis=1)]
            user_data_yearly = user_data_yearly.loc[user_data_yearly.sum(axis=1) >= 4]

            if len(user_data_yearly) >= window_size:
                data_id[id] = user_data_yearly

    # calculate the sum of each column
    # column_sums = concatenated.sum()

    selected_columns = []
    for id in data_id:
        col_sum = data_id[id].sum(axis=0)
        selected_columns.extend(list(col_sum[col_sum >= 1].index)) #6
    selected_columns = list(set(selected_columns))

    for id in data_id:
        data_id[id] = data_id[id][selected_columns]
    ##################################################################################"""

    """merge the users data to train LSTM model"""

    new_data_id = {}
    # loop through users and select the best R rows
    for user, data in data_id.items():

        rows = len(data)

        if rows == window_size:
            new_data_id[user] = data
        elif rows > window_size:
            # get sum of consecutive rows and select best R rows
            sums = [np.array(data)[i, :].sum() for i in range(data.shape[0])]
            best_rows = np.sort(np.argsort(sums)[-window_size:])  #last_N_elements = [5, 5, 3, 3, 4, 3][-N:]
            new_da = data.iloc[best_rows]
            if len(new_da)==window_size:
                new_data_id[user]=new_da

            # check shape of new data for each user
    for user in new_data_id:
        print(user, new_data_id[user].shape)
        if new_data_id[user].shape[0] <window_size:
            print(new_data_id[user])

    data_id=new_data_id
    # concatenate all dataframes in data_id dictionary
    new_data_id = pd.concat(new_data_id, axis=0, ignore_index=True)

    ##################################################################################
    new_data_id = pd.concat(data_id, axis=0, ignore_index=True)
    ##################################################################################

    data_i = np.array(new_data_id)
    data_i = pd.DataFrame(data_i)
    data_i.to_csv("new_data_id100ML.csv")
    #data_i.to_csv("new_data_id27ML.csv")

    new_data_id1 = pd.read_csv(
        "new_data_id100ML.csv",
        index_col=0, delimiter=',')
    new_data_id1 = pd.DataFrame(new_data_id1)

    new_data_id2 = pd.read_csv(
        "new_data_id27ML.csv",
        index_col=0, delimiter=',')
    new_data_id2 = pd.DataFrame(new_data_id2)

    new_data_id12 = [pd.DataFrame(new_data_id2), pd.DataFrame(new_data_id1)]



    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(1, 2, 1)
    ax_3 = fig3.add_subplot(1, 2, 2)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1, 2, 1)
    ax_4 = fig4.add_subplot(1, 2, 2)

    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(1, 2, 1)
    ax_5 = fig5.add_subplot(1, 2, 2)

    train_size = int(len(new_data_id1) * 0.95)
    val_size = int((train_size) * 0.25)
    train_data = new_data_id1.iloc[:train_size - val_size, :]
    val_data = new_data_id1.iloc[train_size - val_size:train_size, :]
    test_data = np.array(new_data_id1.iloc[train_size:, :])

    print(test_data)

    for n, new_data_id in enumerate(new_data_id12):
        ##################################################################################
        train_data = new_data_id
        ##################################################################################
        print(train_data)

        # scale data  We'll then scale the data to values between 0 and 1:
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)

        time_steps = window_size  # len(train_data_scaled)
        X_train, y_train = create_sequences(np.array(train_data_scaled), time_steps)

        print("X_train, y_train")
        print(X_train, y_train)

        # X_val, y_val = create_sequences(np.array(val_data_scaled), time_steps)

        # create input sequences and output labels for testing set
        X_test, y_test = create_sequences(np.array(test_data_scaled), time_steps)

        # define model architecture
        # create the early stopping callback

        model_lf = Sequential()
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model_lf.add(
            LSTM(units=32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                 activation='softmax',
                 go_backwards=False))  # model.add(LSTM(32, input_shape=(timesteps, input_dim)))
        # model_lf.add(LSTM(32, return_sequences=True, activation='softmax'))
        model_lf.add(LSTM(16, return_sequences=False, activation='softmax'))
        model_lf.add(Dense(new_data_id.shape[1], activation='softmax'))
        # compile model
        # model.compile(loss='mean_squared_error', optimizer='adam')
        model_lf.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # train model
        history_lf = model_lf.fit(X_train, y_train, epochs=500, batch_size=30, verbose=1)

        y_pred_lf = model_lf.predict(X_test)


        model_bf = Sequential()
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model_bf.add(LSTM(units=32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                          activation='softmax', go_backwards=True))
        # model_bf.add(LSTM(128, return_sequences=True, activation='softmax', go_backwards=True))
        # model_bf.add(LSTM(32, return_sequences=True, activation='softmax', go_backwards=True))
        model_bf.add(LSTM(16, return_sequences=False, activation='softmax', go_backwards=True))
        model_bf.add(Dense(new_data_id.shape[1], activation='softmax'))

        # compile model
        model_bf.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # train model
        history_bf = model_bf.fit(X_train, y_train, epochs=500, batch_size=30, verbose=1)

        y_pred_bf = model_lf.predict(X_test[:, ::-1, :])

        model_bilstm = Sequential()
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model_bilstm.add(Bidirectional(
            LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                 activation='softmax')))
        # model_bilstm.add(Bidirectional(LSTM(128, return_sequences=True, activation='softmax')))
        model_bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation='softmax')))
        model_bilstm.add(Bidirectional(LSTM(16, return_sequences=False, activation='softmax')))
        model_bilstm.add(Dense(new_data_id.shape[1], activation='softmax'))
        model_bilstm.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        history_bilstm = model_bilstm.fit(X_train, y_train, epochs=500, batch_size=30, verbose=1)

        y_pred_bilstm = model_bilstm.predict(X_test)

        if n == 0:
            ax3.plot(savgol_filter(np.array(history_lf.history['accuracy']), 5, 1))
            ax4.plot(history_lf.history['loss'])
            ax4.plot(history_bf.history['loss'])
            ax3.set_title('Model accuracy')
            ax3.set_ylabel('Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.legend(['Train LSTM Forward', 'Train LSTM Backward', 'Train BiLSTM'], loc='upper left')

            # Plot training & validation loss values
            ax4.plot(history_bilstm.history['loss'])
            # ax4.plot(history_bilstm.history['val_loss'])
            ax4.set_title('Model loss')
            ax4.set_ylabel('Loss')
            ax4.set_xlabel('Epoch')
            ax4.legend(['Train LSTM Forward', 'Train LSTM Backward', 'Train BiLSTM'], loc='upper right')

            ax5.plot(np.arange(len(y_pred_bilstm[-1])), y_pred_bilstm[-1], linestyle='-', marker='*', markersize=10,
                     label='BiLSTM')
            ax5.plot(np.arange(len(y_pred_bf[-1])), y_pred_bf[-1], linestyle='-.', marker='*', markersize=10,
                     label='LSTM Backward')
            ax5.plot(np.arange(len(y_pred_lf[-1])), y_pred_lf[-1], linestyle='--', marker='*', markersize=10,
                     label='LSTM Foreward')
            ax5.plot(np.arange(len(y_test[-1])), y_test[-1], linestyle='--', marker='*', markersize=10,
                     label='Original')
            ax5.set_xticks(range(0, len(new_data_id.columns)), new_data_id.columns, rotation=30)
            # ax5.set_yticks()
            # plt.xlabel('File', fontsize='12', labelpad=15)  # (watt)  descend_order_idx
            ax5.set_ylabel('Predicted File Demands per UE')
        elif n == 1:
            ax_3.plot(savgol_filter(np.array(history_lf.history['accuracy']), 5, 1))
            ax_4.plot(history_lf.history['loss'])
            ax_4.plot(history_bf.history['loss'])
            ax_3.set_title('Model accuracy')
            ax_3.set_ylabel('Accuracy')
            ax_3.set_xlabel('Epoch')
            ax_3.legend(['Train LSTM Forward', 'Train LSTM Backward', 'Train BiLSTM'], loc='upper left')

            # Plot training & validation loss values
            ax_4.plot(history_bilstm.history['loss'])
            # ax4.plot(history_bilstm.history['val_loss'])
            ax_4.set_title('Model loss')
            ax_4.set_ylabel('Loss')
            ax_4.set_xlabel('Epoch')
            ax_4.legend(['Train LSTM Forward', 'Train LSTM Backward', 'Train BiLSTM'], loc='upper right')

            ax_5.plot(np.arange(len(y_pred_bilstm[-1])), y_pred_bilstm[-1], linestyle='-', marker='*', markersize=10,
                      label='BiLSTM')
            ax_5.plot(np.arange(len(y_pred_bf[-1])), y_pred_bf[-1], linestyle='-.', marker='*', markersize=10,
                      label='LSTM Backward')
            ax_5.plot(np.arange(len(y_pred_lf[-1])), y_pred_lf[-1], linestyle='--', marker='*', markersize=10,
                      label='LSTM Foreward')
            ax_5.plot(np.arange(len(y_test[-1])), y_test[-1], linestyle='--', marker='*', markersize=10,
                      label='Original')
            ax_5.set_xticks(range(0, len(new_data_id.columns)), new_data_id.columns, rotation=30)
            # ax_5.set_yticks()
            # plt.xlabel('File', fontsize='12', labelpad=15)  # (watt)  descend_order_idx
            ax_5.set_ylabel('Predicted File Demands per UE')

    plt.legend()
    plt.show()

    # LOSS GRAPH PLOT
    loss = pd.read_excel(
        r'Loss_All.xlsx',
        sheet_name='Sheet1')



    file_directory = "LSTM_backward_" + str(learning_rate) + "_.h5"
    model_bf.save(file_directory)

    file_directory = "LSTM_Forward_" + str(learning_rate) + "_.h5"
    model_lf.save(file_directory)

    file_directory = "Save_Model/BiLSTM_" + str(learning_rate) + "_.h5"
    model_bilstm.save(file_directory)

    return model_lf, model_bf, model_bilstm, data_id, history_lf, history_bf, history_bilstm