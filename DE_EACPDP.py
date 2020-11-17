import os

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from scipy.optimize import differential_evolution

import config
from function.evaluation_indicator import costEffort_popt, calculate_FPA
from function.load_data import load_data


def objective_function(x, x_train, y_density_train):
    fi = np.dot(x_train[:, 1:], x[1:]) + x[0]
    predict_train = 1 / (1 + np.exp(-fi))
    predict_train[predict_train >= 0.5] = 1
    predict_train[predict_train < 0.5] = 0
    predict_density_train = predict_train / (x_train[:, 0] + 1)
    predicted_density_density_train = np.hstack((np.array([predict_density_train]).T, np.array([y_density_train]).T))
    fpa = calculate_FPA(predicted_density_density_train)
    return -fpa


def DE_EACPDP(dataset_name, repeat):
    # get test dataset
    readPath = '.\\data\\' + dataset_name + '\\'
    writePath = '.\\result\\' + dataset_name + '\\' + dataset_name + '_' + 'DE_EACPDP'
    writeSuffix = str(config.repeats) + '_' + str(config.running_id) + '.csv'
    files = os.listdir(readPath)
    n = len(files)
    for i in range(n):
        x_test, y_test = load_data(readPath + files[i], dataset_name)
        list_costEffort20p = []
        list_costEffort1000 = []
        list_costEffort2000 = []
        list_popt = []
        for j in range(n):
            # 不能让PROMISE相同项目的不同版本间执行项目内预测
            if dataset_name == 'PROMISE':
                target_name = files[i].split('-')[1]
                source_name = files[j].split('-')[1]
                if target_name == source_name:
                    print(str(j + 1) + ' x ' + str(i + 1) + '    ' + 'projectNum = ' + str(
                        n) + '    ' + 'repeat = ' + str(repeat + 1) + ' / ' + str(config.repeats) + '    ' + files[
                              j] + ' x ' + files[i])
                    continue
            if j != i:
                print(str(j + 1) + ' => ' + str(i + 1) + '    ' + 'projectNum = ' + str(
                    n) + '    ' + 'repeat = ' + str(repeat + 1) + ' / ' + str(config.repeats) + '    ' + files[
                          j] + ' => ' +
                      files[i])
                # get train dataset
                x_train, y_train = load_data(readPath + files[j], dataset_name)
                # undersampling
                rus = RandomUnderSampler()
                x_train, y_train = rus.fit_resample(x_train, y_train)
                # generate model
                loc_train = x_train[:, 0]
                y_density_train = y_train / (loc_train + 1)
                bounds = []  # 设置范围
                b = (-10000, 10000)
                for k in range(0, len(x_train[0])):
                    bounds.append(b)
                results = differential_evolution(objective_function, bounds, args=(x_train, y_density_train))
                # predict
                x = results.x
                # fi = np.dot(x_test[:, 0:-1], x[0:-1]) + x[-1]
                fi = np.dot(x_test[:, 1:], x[1:]) + x[0]
                predict_test = 1 / (1 + np.exp(-fi))
                predict_test[predict_test >= 0.5] = 1
                predict_test[predict_test < 0.5] = 0
                # evaluate
                loc_test = x_test[:, 0]
                predict_density_test = predict_test / (loc_test + 1)
                y_density_test = y_test / (loc_test + 1)
                parameter = np.hstack(
                    (np.array([y_density_test]).T, np.array([loc_test]).T, np.array([y_test]).T, np.array([predict_density_test]).T))
                costEffort20p, costEffort1000, costEffort2000, popt = costEffort_popt(parameter)
                costEffort20p = format(costEffort20p, '.3f')
                costEffort1000 = format(costEffort1000, '.3f')
                costEffort2000 = format(costEffort2000, '.3f')
                popt = format(popt, '.3f')
                list_costEffort20p.append(costEffort20p)
                list_costEffort1000.append(costEffort1000)
                list_costEffort2000.append(costEffort2000)
                list_popt.append(popt)
            else:
                print(str(j + 1) + ' x ' + str(i + 1) + '    ' + 'projectNum = ' + str(n) + '    ' + 'repeat = ' + str(
                    repeat + 1) + ' / ' + str(config.repeats) + '    ' + files[j] + ' x ' +
                      files[i])
        df_costEffort20p = pd.DataFrame(list_costEffort20p)
        df_costEffort1000 = pd.DataFrame(list_costEffort1000)
        df_costEffort2000 = pd.DataFrame(list_costEffort2000)
        df_popt = pd.DataFrame(list_popt)
        print('正在写入...')
        df_costEffort20p.to_csv(writePath + '_costEffort20p_' + writeSuffix, mode='a', index=False, header=False)
        df_costEffort1000.to_csv(writePath + '_costEffort1000_' + writeSuffix, mode='a', index=False, header=False)
        df_costEffort2000.to_csv(writePath + '_costEffort2000_' + writeSuffix, mode='a', index=False, header=False)
        df_popt.to_csv(writePath + '_popt_' + writeSuffix, mode='a', index=False, header=False)
        print('写入完毕')
