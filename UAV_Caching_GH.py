import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import Power_SucAllocLag as ps_lag
from pyomo.environ import *
import cvxpy as cp
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
from scipy.interpolate import interp1d
from pandas import DataFrame
from random_words import RandomWords
import math as mt
import math
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

np.set_printoptions(precision=10)
from scipy import special
from keras.models import model_from_json
import matplotlib.ticker as mtick
from matplotlib.pyplot import subplots, show
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cvx
import dmcp
from scipy.interpolate import interp1d
from dmcp.fix import fix
from dmcp.find_set import find_minimal_sets
from dmcp.bcd import is_dmcp

# We use seed to reproduce the same results
np.random.seed(1000)
from scipy.interpolate import make_interp_spline, BSpline
import warnings

warnings.filterwarnings("ignore")
# from time import time

np.random.seed(100)
# Loading the dataset
# start_learning_time = time.time()
import matlab.engine

# set default parameters
plt.rcParams.update({
    'font.size': 19,
    'axes.labelsize': 19,
    'axes.titlesize': 19,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.figsize': [10, 8],  # width, height in inches
    'axes.grid': False,
    'lines.linewidth': 4,
    'lines.markersize': 13,
    'figure.subplot.wspace': 0.5,
})


def calculate_hit_rate(topk_movies_per_group, testing_data):
    hit_count = 0
    for i in range(testing_data.shape[0]):
        group_id = testing_data[i, 0] // num_users_per_group
        temp_topk_movies = topk_movies_per_group[group_id]
        # temp_topk_movies = user_rec[group_id]
        if testing_data[i, 1] in temp_topk_movies:
            hit_count += 1
    hite_rate = (hit_count / testing_data.shape[0]) * 100
    return hite_rate


def optional_user_demand_design():
    ##########################################################################
    # Define the number of files, users, and iterations
    # select the repeated files for the first 5 users
    # Define the users and their file demands
    user_demands_popularity = []
    user_demands_lru = []
    user_demands_lfu = []

    # append index of popular files
    user_demands_popularity.append(
        [np.array(ues_req_j)[:, 2][i] for i, p in enumerate(np.array(ues_req_j)[:, 3]) if p > 0.6])

    # Randomly select a file that has not been chosen for the remaining users
    unused_file_indices = np.setdiff1d(np.arange(args.F), user_demands_popularity)
    user_demands_popularity.append(np.random.choice(unused_file_indices))

    # append index of popular files
    # Get unique elements and their counts
    unique, counts = np.unique(np.array(ues_req_j)[:, 2], return_counts=True)
    for u, c in zip(unique, counts):
        if c > 2:
            user_demands_lru.append(u)

    # Randomly select a file that has not been chosen for the remaining users
    unused_file_indices = np.setdiff1d(np.arange(args.F), user_demands_lru)
    user_demands_lru.append(np.random.choice(unused_file_indices))

    print(user_demands_popularity)
    print(user_demands_lru)


def updateReqStatistic(args):
    '''[10] Content Request Statistic of each UE'''
    for u in range(UE):
        reqStatistic[u][Req[u]] += 1
    '''[15] Content request profile similarity'''
    # print('LA.norm(self.reqStatistic, axis=1)=',LA.norm(self.reqStatistic, axis=1))
    reqStatistic_norm = reqStatistic / (linalg.norm(reqStatistic, axis=1)).reshape((UE, 1))
    ueSimilarity = np.matmul(reqStatistic_norm, reqStatistic_norm.T)


def get_user_preference(args, UE_req_predict):
    nUE = args.I
    F = args.F
    UE_req_predict_Bi = UE_req_predict  # args.UE_gamma_pred

    # User Preference is a score list of for each file. Score 0 is the most favorite.
    # i.e. userPreference[0] = [3 2 0 1 4], the most favorate file of UE0 is 2th file, the second favorite file is 3th file
    userPreference = np.zeros((nUE, F), dtype=int)
    for u in range(nUE):
        seedPreference = UE_req_predict_Bi[u]
        userPreference[u] = [sorted(range(len(seedPreference)), key=lambda i: seedPreference[i], reverse=True).index(i)
                             for i in range(len(seedPreference))]
    return userPreference


def get_user_requests(args, omega_k, UE_req_predict):
    user_preference = get_user_preference(args, UE_req_predict)
    F = args.F
    UE_req_predict_Bi = UE_req_predict  # args.UE_gamma_pred
    requests = []
    for j, cluster in enumerate(omega_k):
        req = []
        nUE_j = len(cluster)
        Req_j = [F] * nUE_j
        for i, u in enumerate(cluster):
            Req_j[i] = list(user_preference[u]).index(0)  # index of file 0 (most prefered)
            req.append([j, u, Req_j[i], UE_req_predict_Bi[u][Req_j[i]]])
        requests.append(req)
    return requests


def compute_percentage_score_of_files(args, UE_req_predict_Bi):
    F = args.F
    content_items = []
    for f in range(F):
        f_profile = {'f_id': 0, 'f_size': 0, 'popularity': 0, 'frequency': 0, 'timestamp': 0}  # item[]
        f_profile['f_id'] = f
        f_profile['f_size'] = args.eta_f[f]
        f_profile['popularity'] = sum(UE_req_predict_Bi[:, f]) / F  # alpha*(UE_req_predict_Bi[:,f]), alpha=1/(N-1)
        content_items.append(f_profile)
    return content_items


def get_available_storage(args, uav_cache, total_storage):
    print(uav_cache)
    if len(uav_cache['f_id']) != 0:
        return total_storage - len(uav_cache['f_id']) * args.eta_f
    else:
        return total_storage


###########################################
# M_G_1_Queue:

########################################
def get_queuing_delay(args, user_id, uav_id, queue_len):
    # queue_len = args.uav_queue_length[uav_id]
    rho_j = args.lamda / args.mu
    if rho_j >= 1:
        raise ValueError("UAV queue is overloaded.")
    else:
        expected_num_requests = args.lamda / (args.mu - args.lamda)
        expected_queue_delay = expected_num_requests / args.lamda
        # queue_delay = max(0, expected_queue_delay - (queue_len / lambda_j))
        queue_delay = max(0, (expected_queue_delay - (queue_len / args.lamda)))
        # additional_delay = random.expovariate(lambda_j)
        return queue_delay


# ************************************************************************************

# Define a function to update the UAV cache based on content popularity
def update_uav_cache(args, uav_cache, UAV_CAPACITY):
    # Sort the content items by popularity score in descending order
    sorted_items = sorted(args.content_items, key=lambda item: item['popularity'], reverse=True)
    # Iterate over the sorted content items and add them to the UAV cache until it is full
    for item in sorted_items:
        if len(uav_cache) < UAV_CAPACITY and item not in uav_cache:
            uav_cache.append(item)
    # return uav_cache


# Define a function to handle user requests for content items
def handle_user_request(args, uav_cache, file_id, uav_id, user_index, i_inx, UAV_CAPACITY, rate_k):
    # Check if the requested item is in the UAV cache
    D_i = 0
    fetched_f_uav = 0
    fetched_f_server = 0
    uav_fetch_count = 0
    item_id_list = []

    # transmission delay of the request item for the requesting user
    delay_i = args.eta_f[file_id] / rate_k[i_inx]

    for item in uav_cache:
        item_id_list.append(item['f_id'])

    if file_id in item_id_list:

        D_i += get_queuing_delay(args, user_index, uav_id, args.uav_queue_length[uav_id]) + delay_i
        fetched_f_uav += args.eta_f[file_id]
        # time.sleep(2)

    else:
        # Check if the UAV cache is full
        sum_f_size = 0
        for it in uav_cache:
            sum_f_size += it['f_size']

        while (sum_f_size + args.eta_f[file_id]) >= UAV_CAPACITY:

            # If the UAV cache is full, randomly replace an item with a lower popularity score
            popularity = []
            for item in uav_cache:
                popularity.append(item['popularity'])
            print(popularity)
            lower_popularity_items = np.argmin(popularity)
            if lower_popularity_items is not None:
                uav_cache.remove(uav_cache[lower_popularity_items])
            else:
                uav_cache.pop()

            # check uav cache available size
            sum_f_size = 0
            for it in uav_cache:
                sum_f_size += it['f_size']

        uav_cache.append(args.content_items[file_id])
        # uav_cache.append(args.eta_f[file_id])
        D_i += 0.3 + get_queuing_delay(args, user_index, uav_id, args.uav_queue_length[uav_id]) + delay_i
        fetched_f_server += args.eta_f[file_id]
        args.uav_queue_length[uav_id] += 1

    return D_i, fetched_f_uav, fetched_f_server, uav_cache


def handle_user_request_lfu(args, uav_cache, file_id, uav_id, user_index, i_inx, UAV_CAPACITY, rate_k):
    # Check if the requested item is in the UAV cache
    D_i = 0
    fetched_f_uav = 0
    fetched_f_server = 0
    item_id_list = []

    # Transmission delay of the request item for the requesting user
    delay_i = args.eta_f[file_id] / rate_k[i_inx]

    time.sleep(0.2)  # Adjust the sleep duration as needed

    # Record the timestamp after the sleep
    timestamp = time.perf_counter()

    for item in uav_cache:
        item_id_list.append(item['f_id'])
        # Update the frequency of the requested item by incrementing its count
        if item['f_id'] == file_id:
            item['frequency'] += 1
            item['timestamp'] = timestamp
    if file_id in item_id_list:
        D_i += get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_lfu[uav_id]) + delay_i
        fetched_f_uav += args.eta_f[file_id]
        # time.sleep(2)

    else:
        # Check if the UAV cache is full
        sum_f_size = 0
        for it in uav_cache:
            sum_f_size += it['f_size']

        while (sum_f_size + args.eta_f[file_id]) >= UAV_CAPACITY:

            min_frequency = min(item['frequency'] for item in uav_cache)
            lfu_items = [item for item in uav_cache if item['frequency'] == min_frequency]
            if len(lfu_items) > 1:
                to_remove = min(lfu_items, key=lambda x: x['timestamp'])
            else:
                to_remove = lfu_items[0]
            uav_cache.remove(to_remove)

            # Check UAV cache available size
            sum_f_size = 0
            for it in uav_cache:
                sum_f_size += it['f_size']

        # Add the requested item to the end of the UAV cache with an initial frequency of 1
        for f in args.content_items:
            f['frequency'] += 1
            f['timestamp'] = timestamp
        uav_cache.append(args.content_items[file_id])  # lfu_items, key=lambda x: x['timestamp']
        D_i += 0.3 + delay_i + get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_lfu[uav_id])
        fetched_f_server += args.eta_f[file_id]
        args.uav_queue_length_lfu[uav_id] += 1

    return D_i, fetched_f_uav, fetched_f_server, uav_cache


# Define a function to handle user requests for content items using the LRU cache policy
def handle_user_request_lru(args, uav_cache, file_id, uav_id, user_index, i_inx, UAV_CAPACITY,
                            rate_k):  # item_name, uav_cache

    # Check if the requested item is in the UAV cache
    D_i = 0
    fetched_f_uav = 0
    fetched_f_server = 0
    item_id_list = []

    # transmission delay of the request item for the requesting user
    delay_i = args.eta_f[file_id] / rate_k[i_inx]
    time.sleep(0.2)  # Adjust the sleep duration as needed

    # Record the timestamp after the sleep
    timestamp = time.perf_counter()

    for item in uav_cache:
        item_id_list.append(item['f_id'])
        if item['f_id'] == file_id:  # """if item.file_id == file_id:
            item['timestamp'] = timestamp  # item.timestamp = time()"""

    # Check if the requested item is in the UAV cache  args.UE_Latencies_Net[uav_id][i_inx] +
    if file_id in item_id_list:
        D_i += get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_lru[uav_id]) + delay_i
        # Move the requested item to the end of the list to update its position in the LRU cache
        fetched_f_uav += args.eta_f[file_id]
        # time.sleep(2)


    else:

        # Check if the UAV cache is full
        sum_f_size = 0
        for it in uav_cache:
            sum_f_size += it['f_size']
        while (sum_f_size + args.eta_f[file_id]) >= UAV_CAPACITY:
            # If the UAV cache is full, remove the least recently used item from the cache
            # to_remove = min(uav_cache, key=lambda x: x.timestamp)
            to_remove = min(uav_cache, key=lambda x: x['timestamp'])
            uav_cache.remove(to_remove)

            # to_remove = uav_cache.pop(0)

            # check uav cache available size
            sum_f_size = 0
            for it in uav_cache:
                sum_f_size += it['f_size']
        # Add the requested item to the end of the UAV cache with an initial frequency of 1
        for f in args.content_items:
            f['timestamp'] = timestamp
        # Add the requested item to the end of the UAV cache
        uav_cache.append(args.content_items[file_id])
        D_i += 0.3 + delay_i + get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_lru[uav_id])
        # args.uav_queue_length[uav_id] += 1
        fetched_f_server += args.eta_f[file_id]
        args.uav_queue_length_lru[uav_id] += 1
    return D_i, fetched_f_uav, fetched_f_server, uav_cache


# Define a function to handle user requests for content items using the FIFO cache policy
def handle_user_request_fifo(args, uav_cache, file_id, uav_id, user_index, i_inx, UAV_CAPACITY,
                             rate_k):  # item_name, uav_cache
    # Check if the requested item is in the UAV cache
    # Check if the requested item is in the UAV cache
    D_i = 0
    fetched_f_uav = 0
    fetched_f_server = 0
    item_id_list = []

    # transmission delay of the request item for the requesting user
    delay_i = args.eta_f[file_id] / rate_k[i_inx]

    for item in uav_cache:
        item_id_list.append(item['f_id'])
    # Check if the requested item is in the UAV cache  args.UE_Latencies_Net[uav_id][i_inx] +
    if file_id in item_id_list:
        D_i += get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_fifo[uav_id]) + delay_i
        # Move the requested item to the end of the list to update its position in the LRU cache
        fetched_f_uav += args.eta_f[file_id]
        # time.sleep(2)

    else:
        # print(f'{file_id} not found in UAV cache')
        # Check if the UAV cache is full
        sum_f_size = 0
        for it in uav_cache:
            sum_f_size += it['f_size']
        while (sum_f_size + args.eta_f[file_id]) >= UAV_CAPACITY:

            to_remove = uav_cache.pop(0)

            # check uav cache available size
            sum_f_size = 0
            for it in uav_cache:
                sum_f_size += it['f_size']

        # Add the requested item to the end of the UAV cache
        D_i += 0.3 + delay_i + get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_fifo[uav_id])
        # args.uav_queue_length[uav_id] += 1
        fetched_f_server += args.eta_f[file_id]
        # fetched_f_uav += args.eta_f
        uav_cache.append(args.content_items[file_id])
        args.uav_queue_length_fifo[uav_id] += 1
    return D_i, fetched_f_uav, fetched_f_server, uav_cache


def handle_user_request_rand(args, uav_cache, file_id, uav_id, user_index, i_inx, UAV_CAPACITY, rate_k):
    # Check if the requested item is in the UAV cache
    D_i = 0
    fetched_f_uav = 0
    fetched_f_server = 0
    uav_fetch_count = 0
    item_id_list = []

    # transmission delay of the request item for the requesting user
    delay_i = args.eta_f[file_id] / rate_k[i_inx]

    for item in uav_cache:
        item_id_list.append(item['f_id'])

    if file_id in item_id_list:
        D_i += get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_rand[uav_id]) + delay_i
        fetched_f_uav += args.eta_f[file_id]
        # time.sleep(2)

    else:
        # Check if the UAV cache is full
        sum_f_size = 0
        for it in uav_cache:
            sum_f_size += it['f_size']
        while (sum_f_size + args.eta_f[file_id]) >= UAV_CAPACITY:
            # If the UAV cache is full, randomly replace an item with a lower popularity score

            index = np.random.randint(0, len(uav_cache))
            to_remove = uav_cache.pop(index)

            # check uav cache available size
            sum_f_size = 0
            for it in uav_cache:
                sum_f_size += it['f_size']
        uav_cache.append(args.content_items[file_id])
        D_i += 0.3 + get_queuing_delay(args, user_index, uav_id, args.uav_queue_length_rand[uav_id]) + delay_i
        fetched_f_server += args.eta_f[file_id]
        # fetched_f_uav += args.eta_f
        args.uav_queue_length_rand[uav_id] += 1

    return D_i, fetched_f_uav, fetched_f_server, uav_cache