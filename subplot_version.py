import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import re


def linear_function(x, a, b):
    return a * x + b


def measurement_error(data, last_digit_error, percentage_error):
    return ((data * percentage_error / 100) ** 2 + last_digit_error ** 2) ** (
            1 / 2)


def characterize_fit(y, prediction, uncertainty, param_num):
    return np.sum(((y - prediction) / uncertainty) ** 2) / (len(y) - param_num)


def plot_volt_vs_curr(current, current_uncertainty, voltage, voltage_uncertainty, graph_name):
    plt.title("Voltage at each Current For {}".format(graph_name))
    plt.ylabel("Voltage(V)")
    plt.xlabel("Current(mA)")
    plt.errorbar(current, voltage, xerr=current_uncertainty,
                 yerr=voltage_uncertainty, ls='', lw=1, marker='o', markersize=2,
                 label="figs/{}data with error bar".format(graph_name))
    popt, pcov = curve_fit(linear_function, current, voltage, sigma=voltage_uncertainty,
                           absolute_sigma=True)
    voltage_prediction = linear_function(current, *popt)
    plt.plot(current, voltage_prediction, label="best fit line")
    # plt.legend()
    print("{} resistance:{}".format(graph_name, -popt[0]))
    return voltage_prediction


def plot_residual(x, y, uncertainty, prediction, graph_name):
    plt.plot(x, np.zeros_like(y), "g-", label="Z = 0")
    plt.errorbar(x, y - prediction, yerr=uncertainty, ls='', lw=1, marker='o', markersize=2, label="residual")
    print("{} chi_sq = {}".format(graph_name, characterize_fit(y, prediction, uncertainty, 2)))


def analysis(csv_files_path, data_dir):
    plot_name = re.sub(', Option [1-2]|\.csv', "", csv_files_path[0]). replace(".", " point ")
    print(plot_name)
    plt.figure(plot_name)
    for i in range(len(csv_files_path)):
        subplot_name = csv_files_path[i].replace(".csv", "")
        volt_data, curr_data, volt_measurement_error, curr_measurement_error = process_data_set(os.path.join(data_dir, csv_files_path[i]))
        print(volt_data, curr_data, volt_measurement_error, curr_measurement_error)
        plt.subplot(2, 2, i*2 + 1)
        prediction = plot_volt_vs_curr(curr_data, curr_measurement_error, volt_data, volt_measurement_error,
                                       subplot_name)
        plt.subplot(2, 2, i*2 + 2)
        plot_residual(curr_data, volt_data, volt_measurement_error, prediction, subplot_name)
        plt.tight_layout()
    plt.savefig("figs/" + plot_name)


def process_data_set(path):
    # loading the data from the csv file
    volt_data, curr_data, volt_lastdig_err, volt_per_error, curr_lastdig_err, curr_per_err = np.loadtxt(
        path, skiprows=1,
        delimiter=',', unpack=True, usecols=(1, 0, 6, 5, 4, 3))
    print(curr_data, volt_data, curr_lastdig_err, volt_lastdig_err)
    curr_measurement_error = measurement_error(curr_data, curr_lastdig_err,
                                               curr_per_err)
    print(curr_measurement_error)
    volt_measurement_error = measurement_error(volt_data, volt_lastdig_err,
                                               volt_per_error)
    return volt_data, curr_data, volt_measurement_error, curr_measurement_error


def group_data_set(unsorted_list):
    groups = {}
    for i in unsorted_list:
        key = i.split()[-1]
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    groups_copy = groups.copy()
    groups_copy["battery"] = []
    for i in groups:
        if len(groups[i]) == 1:
            groups_copy["battery"].append(groups[i][0])
            groups_copy.pop(i)
    return groups_copy


if __name__ == "__main__":
    # run on every file in \data
    directory = 'data'
    grouped_data = group_data_set(os.listdir(directory))
    print(grouped_data)
    for j in grouped_data:
        analysis(grouped_data[j], directory)
