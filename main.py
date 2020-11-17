import time

import pandas as pd

import config
from DE_EACPDP import DE_EACPDP
from matchMetrics.NASAMatchMetrics import NASAMatchMetrics
from matchMetrics.RELINKMatchMetrics import RELINKMatchMetrics

dataset_name = config.datasets[config.current_dataset]

# 对NASA和RELINK进行指标匹配
if dataset_name == 'NASA':
    NASAMatchMetrics()
if dataset_name == 'RELINK':
    RELINKMatchMetrics()

time_start = time.time()
for repeat in range(config.repeats):
    DE_EACPDP(dataset_name, repeat)
time_stop = time.time()
time_list = []
time_list.append(time_stop - time_start)
df_time = pd.DataFrame(time_list)
writePath = '.\\result\\' + dataset_name + '\\' + dataset_name + '_' + 'DE_EACPDP'
writeSuffix = str(config.repeats) + '_' + str(config.running_id) + '.csv'
df_time.to_csv(writePath + '_time_' + writeSuffix, mode='a', index=False, header=False)
