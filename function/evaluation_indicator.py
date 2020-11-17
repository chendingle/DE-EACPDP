import numpy as np


def get_worst_optimal_area(data):
    """
    :param data: density_effort_defect_predictDensity
    :return: worst area, optimal area
    """
    total_effort = np.sum(data[:, 1])  # 求loc列的和
    total_defect = np.sum(data[:, 2])  # 求y_test的和
    data = data[data[:, 0].argsort()]  # 根据测试集的density排序

    # calculate actual worst area
    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    for i in range(data.shape[0]):  # 测试集的个数
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    worst_area = np.trapz(point_y, point_x)

    # calculate actual optimal area
    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    i = data.shape[0] - 1
    while i >= 0:
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
        i = i - 1
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    optimal_area = np.trapz(point_y, point_x)

    return worst_area, optimal_area


def costEffort_popt(data):
    """
    :param data: density_effort_defect_predictDensity
    :return: ACC, Popt
    """
    worst_area, optimal_area = get_worst_optimal_area(data)
    total_instance = data.shape[0]
    total_effort = np.sum(data[:, 1])
    total_defect = np.sum(data[:, 2])
    costEffort20p_mark = False
    costEffort1000_mark = False
    costEffort2000_mark = False
    threshold = 0.2 * total_effort
    data = data[data[:, 3].argsort()]
    # calculate predicted area
    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    i = total_instance - 1
    while i >= 0:
        if costEffort20p_mark is False and x_current > threshold:
            costEffort20p = y_current / total_defect
            costEffort20p_mark = True
        if costEffort1000_mark is False and x_current > 1000:
            costEffort1000 = y_current / total_defect
            costEffort1000_mark = True
        if costEffort2000_mark is False and x_current > 2000:
            costEffort2000 = y_current / total_defect
            costEffort2000_mark = True
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
        i = i - 1
    # 防止项目总loc数不到1000或者不到2000，而报local variable 'costEffort2000' referenced before assignment
    if x_current <= 1000:
        costEffort1000 = y_current / total_defect
        costEffort2000 = y_current / total_defect
    if 1000 < x_current <= 2000:
        costEffort2000 = y_current / total_defect
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    predicted_area = np.trapz(point_y, point_x)
    popt = 1 - (optimal_area - predicted_area) / (optimal_area - worst_area)

    return costEffort20p, costEffort1000, costEffort2000, popt


def calculate_FPA(data):
    """
    :param data: predictDensity_density
    :return: FPA
    """
    data = data[data[:, 0].argsort()]
    sum = 0
    K = data.shape[0]
    for i in range(K):
        sum = sum + (i + 1) * data[i, 1]
    return sum
