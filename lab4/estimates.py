import numpy as np
from read_dir import rawData_instance
import matplotlib.pyplot as plt
from tabulate import tabulate
np.float_ = np.float64
import intvalpy as ip
import scipy.special


def print_table(matrix, headers):
    print(tabulate(matrix, headers, tablefmt="simple_grid", stralign='center'))


def boxplot_T(data_list):
    LQ = np.quantile(data_list, 0.25)
    UQ = np.quantile(data_list, 0.75)
    IQR = UQ - LQ
    s = sorted(data_list)
    x_L = max(s[0], LQ - 3 / 2 * IQR)
    x_U = min(s[-1], UQ + 3 / 2 * IQR)

    return np.array([x_L, x_U])


def checking_for_anomaly(data_list, ranges, print_flag=False):
    idx = []
    A = []
    for i in range(len(data_list)):
        row = [f"x({i + 1})", f"{data_list[i]}"]
        if data_list[i] not in ip.Interval(ranges):
            row.append("аномальна")
            idx.append(i)
            A.append(row)
    if print_flag:
        print_table(A, headers=['x_i', 'Значение', 'Результат'])
    return idx


def remove_outliers(data_list, print_flag):
    ranges = boxplot_T(data_list)
    idx = checking_for_anomaly(data_list, ranges, print_flag)
    data_list = np.delete(data_list, idx)
    return data_list


def get_ynow(ch, cells, fn, print_flag=False):
    CH18 = fn.frames  # 100 штук по 8
    Ch = []
    for i in range(fn.frame_count):
        Chnow = np.array(CH18[i])[:, ch]
        Ch.append(Chnow)
    Ch = np.array(Ch).T  # 100 раз по 1024 транспонируем и получаем 1024 раза по 100
    ynow = Ch[cells, :]  # берем ячейку номер cells (от 0 до 1023) и вытаскиваем соответсвующий массив из 100 элементов
    ynow_cut = remove_outliers(ynow, print_flag)

    ynow_V = [ynow[i] / 16385 - 0.5 for i in range(len(ynow))]
    ynow_cut_V = [ynow_cut[i] / 16385 - 0.5 for i in range(len(ynow_cut))]
    # print(ynow_cut_V)
    width = (np.max(ynow_cut_V) - np.min(ynow_cut_V)) * 1000
    # print(width)

    if print_flag:
        plt.subplot(1, 2, 1)
        plt.hist(ynow_V, edgecolor="cornflowerblue", bins=30)
        plt.subplot(1, 2, 2)
        plt.hist(ynow_cut_V, edgecolor="cornflowerblue", bins=30)
        plt.title(f"{fn.lvl}")
        plt.show()

    return ynow_cut, width


def calibration_data_int(ynow, frame_count):  # получаем внутренние оценки для каждой ячейки каждого из каналов
    ynows = np.sort(ynow)
    ynowint = [ynows[frame_count // 4], ynows[3 * frame_count // 4]]
    return np.array(ynowint)


def calibration_data_ext(ynow):
    ynowint = [np.min(ynow), np.max(ynow)]
    return np.array(ynowint)


def calibration_data_ext2(ynow):
    ynowsort = np.sort(ynow)
    ynowint = [ynowsort[1], ynowsort[-2]]
    return np.array(ynowint)


def calibration_data_by_bin(fn, ch, cells, type="All", print_flag=False):
    yarray = 0
    ynow = get_ynow(ch=ch, cells=cells, fn=fn, print_flag=print_flag)[0]
    match type:
        case "Ext":
            yarray = calibration_data_ext(ynow)
        case "Ext2":
            yarray = calibration_data_ext2(ynow)
        case "Int":
            yarray = calibration_data_int(ynow, fn.frame_count)
        case "All":
            yarray = [calibration_data_int(ynow, fn.frame_count), calibration_data_ext(ynow)]
    return np.array(yarray)


def calibration_data_all_bins(ch, cells, type: str, data, print_flag=False):
    yarray = []
    widths = []
    for fn in data.bins:
        get_ynow_res = get_ynow(ch=ch, cells=cells, fn=fn, print_flag=print_flag)
        ynow = get_ynow_res[0]
        widths.append(get_ynow_res[1])
        match type:
            case "Ext":
                yarray.append(calibration_data_ext(ynow))
            case "Ext2":
                yarray.append(calibration_data_ext2(ynow))
            case "Int":
                yarray.append(calibration_data_int(ynow, fn.frame_count))
    return np.array(yarray)

