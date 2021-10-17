from dtw import *
from scipy.spatial import distance
import gc
import time
import datetime
import copy
from random import *
INF = 2147483647
gc.collect()

def random_window_dtw(length, start_idx, timesteps, k_list, test_data_set, train_data_set2, calculate):
    gc.collect()

    euclid_predict, euclid_real = [], []
    dtw_predict, dtw_real = [], []

    predict_list1, real_value1, running_time1 = [], [], []
    predict_list2, real_value2, running_time2 = [], [], []
    predict_list3, real_value3, running_time3 = [], [], []

    predict_list4, real_value4, running_time4 = [], [], []
    predict_list5, real_value5, running_time5 = [], [], []
    predict_list6, real_value6, running_time6 = [], [], []

    for idx, test in enumerate(test_data_set):
        train_data_set = copy.deepcopy(train_data_set2)
        if idx == 0:
            train_data_set += test_data_set[1:]
            # print(len(train_data_set))
        elif idx == 1:
            train_data_set += test_data_set[0:3:2]
            # print(len(train_data_set))
        elif idx == 2:
            train_data_set += test_data_set[0:2]
            # print(len(train_data_set))
        test_data_length = length[idx]
        test_data = test[::timesteps]
        real_test_data = test_data[start_idx[idx]: start_idx[idx] + test_data_length]
        real = len(test_data) - (start_idx[idx] + test_data_length)
        
        real *= timesteps
        print(f'euclidean motor {idx + 1}의 고장 예측: {start_idx[idx]} ~ {start_idx[idx] + test_data_length}, {len(test)} -> {len(test_data)}')

        ecu_dist_list = []
        idx2 = 0
        starttime = time.time()
        for train_data in train_data_set:
            train_len = test_data_length
            cal = 0
            while cal < calculate:
                i = randint(0, len(train_data) - train_len)
                get_ecu = distance.euclidean(real_test_data, train_data[i:i + train_len])
                remain_time = len(train_data) - (i + train_len)
                ecu_dist_list.append((get_ecu, idx2, remain_time))
                idx2 += 1
                cal += 1

        sort_list = sorted(ecu_dist_list)
        for k in k_list:
            predict = 0
            similarity = []
            for i in range(k):
                dist = sort_list[i][0]
                if dist == 0:
                    similarity.append(INF)
                else:
                    similarity.append(1 / dist)

            for i in range(k):
                predict += (similarity[i] * sort_list[i][2] / sum(similarity))
                
            predict = round(predict, 6)
            euclid_predict.append(predict)
            euclid_real.append(real)
            times_ec = time.time()-starttime
            
            if idx == 0:
                predict_list1.append(predict)
                real_value1.append(real)
                running_time1.append(times_ec)
            elif idx == 1:
                predict_list2.append(predict)
                real_value2.append(real)
                running_time2.append(times_ec)
            else:
                predict_list3.append(predict)
                real_value3.append(real)
                running_time3.append(times_ec)

        print('계산완료') 


    for idx, test in enumerate(test_data_set):
        train_data_set = copy.deepcopy(train_data_set2)
        if idx == 0:
            train_data_set += test_data_set[1:]
            # print(len(train_data_set))
        elif idx == 1:
            train_data_set += test_data_set[0:3:2]
            # print(len(train_data_set))
        elif idx == 2:
            train_data_set += test_data_set[0:2]
        test_data_length = length[idx]
        test_data = test[::timesteps]
        real_test_data = test_data[start_idx[idx]: start_idx[idx] + test_data_length]
        real = len(test_data) - (start_idx[idx] + test_data_length)
        
        real *= timesteps
        print(
            f'dtw motor {idx + 1}의 고장 예측: {start_idx[idx]} ~ {start_idx[idx] + test_data_length}, {len(test)} -> {len(test_data)}')

        dtw_dist_list = []
        idx2 = 0
        starttime2 = time.time()
        for train_data in train_data_set:
            train_len = (test_data_length - 1) * timesteps + 1
            cal = 0
            while cal < calculate:
                i = randint(0, len(train_data) - train_len)
                get_dtw = dtw(real_test_data, train_data[i:i + train_len], keep_internals=True,
                              distance_only=True)
                remain_time = len(train_data) - (i + train_len)
                dtw_dist_list.append((get_dtw.distance, idx2, remain_time))
                idx2 += 1
                cal += 1

        sort_list = sorted(dtw_dist_list)
        for k in k_list:
            predict = 0
            similarity = []
            for i in range(k):
                dist = sort_list[i][0]
                if dist == 0:
                    similarity.append(INF)
                else:
                    similarity.append(1 / dist)

            for i in range(k):
                predict += (similarity[i] * sort_list[i][2] / sum(similarity))
            predict = round(predict, 6)
            dtw_predict.append(predict)
            dtw_real.append(real)
            times_dt = time.time()-starttime2
            if idx == 0:
                predict_list4.append(predict)
                real_value4.append(real)
                running_time4.append(times_dt)
            elif idx == 1:
                predict_list5.append(predict)
                real_value5.append(real)
                running_time5.append(times_dt)
            else:
                predict_list6.append(predict)
                real_value6.append(real)
                running_time6.append(times_dt)
        
        print('계산완료') 
    return predict_list1, predict_list2, predict_list3, predict_list4, predict_list5, predict_list6, \
           real_value1, real_value2, real_value3, real_value4, real_value5, real_value6, euclid_predict, \
               dtw_predict, euclid_real, dtw_real, running_time1, running_time2, running_time3, running_time4, \
                   running_time5, running_time6
           
    # return euclid_predict, dtw_predict, euclid_real, dtw_real

