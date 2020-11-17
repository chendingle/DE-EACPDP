#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def load_data(path, dataset_name):
    if dataset_name in ['AEEEM', 'NASA', 'RELINK']:
        data = pd.read_csv(path).values
        x_data = data[:, 0:-1]
        x_data = x_data.astype(np.float64)
        # 这里不能用reshape(-1, 1)，否则density_train = y_train / (loc_train + 1)这一步会变成方阵
        y_data = data[:, -1]
        y_data = y_data.astype(np.bool)  # 当标记>1时也能按照1统计
        return x_data, y_data
    elif dataset_name == 'PROMISE':
        data = pd.read_csv(path).values
        x_data = np.hstack((data[:, 0].reshape(-1, 1), data[:, 4:-1]))
        x_data = x_data.astype(np.float64)
        # 这里不能用reshape(-1, 1)，否则density_train = y_train / (loc_train + 1)这一步会变成方阵
        y_data = data[:, -1]
        y_data = y_data.astype(np.bool)
        return x_data, y_data
