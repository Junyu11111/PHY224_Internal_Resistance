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


def current_vs_voltage(current_lst, current_uncertainty_lst, voltage_lst, voltage_uncertainty_lst, data_set_name,
                       graph_name):
    plt.title("Current at each Volt For {}".format(graph_name))
    plt.xlabel("Voltage(V)")
    plt.ylabel("Current(mA)")
    current = np.empty((0, 0))
    voltage = np.empty((0, 0))
    voltage_uncertainty = np.empty((0, 0))
    for i in range(len(voltage_lst)):
        plt.errorbar(current_lst[i], voltage_lst[i], xerr=voltage_uncertainty_lst[i], yerr=voltage_uncertainty_lst[i],
                     marker='o', markersize=2, label="{} data with error bar".format(data_set_name))
        current = np.append(current, current_lst[i])
        voltage = np.append(voltage, voltage_lst[i])
        voltage_uncertainty = np.append(voltage_uncertainty, voltage_uncertainty_lst[i])
    popt, pcov = curve_fit(linear_function, current, voltage, sigma=voltage_uncertainty,
                           absolute_sigma=True)
    print("{} resistance:{}".format(graph_name, popt[1]))
    voltage_prediction = linear_function(np.sort(voltage), *popt)
    plt.plot(np.sort(current), voltage_prediction, label="best fit line")
    plt.legend()
    plt.figure("residual")
    plt.plot(current, np.zeros_like(current), "g-", label="Z = 0")
    plt.errorbar(current, voltage - voltage_prediction, yerr=voltage_uncertainty, ls='', lw=1, marker='o', markersize=2,
                 label="{} residual".format(graph_name))
    print("{} chi_sq = {}".format(graph_name, characterize_fit(voltage, voltage_prediction, voltage_uncertainty, 2)))


def battery(path, name):
    volt_data, curr_data, volt_lastdig_err, volt_per_error, curr_lastdig_err, curr_per_err = np.loadtxt(
        path, skiprows=1,
        delimiter=',', unpack=True, usecols=(1, 0, 5, 6, 3, 4))
    print(curr_data, volt_data, curr_lastdig_err, volt_lastdig_err)
    curr_measurement_error = measurement_error(curr_data, curr_lastdig_err,
                                               curr_per_err)
    print(curr_measurement_error)
    volt_measurement_error = measurement_error(volt_data, volt_lastdig_err,
                                               volt_per_error)
    plt.figure(1)
    current_vs_voltage(np.array(curr_data, ndmin=2), np.array(curr_measurement_error, ndmin=2),
                       np.array(volt_data, ndmin=2),
                       np.array(volt_measurement_error, ndmin=2), "battery", name)


if __name__ == "__main__":
    battery("Internal Resistance of the Power Supply Data - Battery, Option 1 (1).csv", "battery option 1")
    plt.show()
