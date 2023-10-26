import os

import numpy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def linear_function(x, a, b):
    return a * x + b


def measurement_error(data, last_digit_error, percentage_error):
    return ((data * percentage_error / 100) ** 2 + last_digit_error ** 2) ** (
            1 / 2)


def characterize_fit(y, prediction, uncertainty, param_num):
    return np.sum(((y - prediction) / uncertainty) ** 2) / (len(y) - param_num)


def analysis(current, current_uncertainty, voltage, voltage_uncertainty,
             graph_name):
    # plotting the volt vs curr graph
    plt.figure("Voltage at each Current For {}".format(graph_name))
    plt.title("Voltage at each Current For {}".format(graph_name))
    plt.ylabel("Voltage(V)")
    plt.xlabel("Current(mA)")
    plt.errorbar(current, voltage, xerr=current_uncertainty,
                 yerr=voltage_uncertainty, ls='', lw=1, marker='o', markersize=2,
                 label="figs/{}data with error bar".format(graph_name))
    popt, pcov = curve_fit(linear_function, current, voltage, sigma=voltage_uncertainty,
                           absolute_sigma=True)
    print("{} resistance:{}".format(graph_name, -popt[1]))
    voltage_prediction = linear_function(current, *popt)
    plt.plot(current, voltage_prediction, label="best fit line")
    plt.legend()
    plt.savefig("figs/Voltage at each Current For {}".format(graph_name).replace(".", " point "))
    # plotting the residual
    plt.figure("{} residual".format(graph_name))
    plt.title("{} residual".format(graph_name))
    plt.ylabel("Voltage(V)")
    plt.xlabel("Current(mA)")
    plt.plot(current, np.zeros_like(current), "g-", label="Z = 0")
    plt.errorbar(current, voltage - voltage_prediction, yerr=voltage_uncertainty, ls='', lw=1, marker='o', markersize=2,
                 label="{} residual".format(graph_name))
    print("{} chi_sq = {}".format(graph_name, characterize_fit(voltage, voltage_prediction, voltage_uncertainty, 2)))
    plt.savefig("figs/{} residual".format(graph_name).replace(".", " point "))


def process_data_set(path, name):
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
    analysis(curr_data, curr_measurement_error, volt_data, volt_measurement_error, name.replace(".csv", ""))


if __name__ == "__main__":
    # run on every file in \data
    directory = 'data'
    for filename in os.listdir(directory):
        process_data_set(os.path.join(directory, filename), filename)
    plt.show()
