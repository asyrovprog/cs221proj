import math, random
import matplotlib.pyplot as plt
import numpy as np

use_daily    = True # use day/night ups and downs
use_weekly   = True # use weekly increases
use_seasonal = True # use seasonal increases
use_trend    = True # use overall increasing trend
use_noise    = True # add some gaussian noise
use_holidays = True # generate fake holiday increase
base_val     = 50.0 # base
holidays     = [0, 2, 16, 40, 120, 124, 230, 250, 340, 350] # simulation of holiday days

# return year for passed hour
def get_year(h):
    return int(int(h) // 24) // 365

# return day of year for passed hour
def get_day_of_year(h):
    return int(int(h) // 24) % 365

# return hour for nearest holiday, that is after passed hour
def get_nearest_holiday_hour(h):
    year = get_year(h)
    for y in range(year, year + 1):
        for i in range(0, len(holidays)):
            holiday_start = (y * 365 + holidays[i]) * 24
            if holiday_start > h:
                return holiday_start
    return h

# transformed math.cos, so it is in range [0..1]
def hill(x):
    return (math.cos(x + math.pi) + 1.0) / 2.0

# this function simulates data, which are in similar way as expected have ups and downs for
# day/night, week days, seasons, holidays (simulated), also add some increasing linear
# trend and noise
def proto_function(h):
    x, daily, weekly, seasonal, trend, noise, holiday = h + 1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    day_hour_period = (2.0 * x * math.pi) / 24.0
    week_hour_period = day_hour_period / 7.0
    seasonal_period = day_hour_period / (365.0 / 4.0)
    doy = get_day_of_year(h)
    if use_daily:
        daily = hill(day_hour_period)                     # daily ups and downs
    if use_weekly:
        v = math.cos(week_hour_period)
        if v > 0:
            weekly = 0.5 * v                              # weekly ups
    if use_seasonal:
        v = math.cos(seasonal_period)
        if v > 0:
            seasonal = 0.5 * math.cos(seasonal_period)    # seasonal downs
    if use_holidays and doy in holidays:
        holiday += daily * 1.5                            # we expect abnormal behaviour on holidays
    if use_trend:
        trend = (base_val / 5e7) * x                      # overall trend
    if use_noise:
        noise = max(0.0, random.gauss(1.0, 0.4))
    return int(0.1 + base_val * (daily + weekly + seasonal + trend + noise + holiday))

def generate_dataset(start, size):
    arr = []
    for i in range(0, size):
        x = i + start
        arr.append(proto_function(i + start))
    return np.array(arr)

def initialize_datasets(normalize):
    train = generate_dataset(0, 750000)
    dev = generate_dataset(750001, 1000000)
    if normalize:
        maxval = max(max(train), max(dev))
        train = np.divide(train, maxval)
        dev = np.divide(train, maxval)
    return dev, train

if __name__ == "__main__":
    def show_holiday_week():
        data = []
        start = (random.uniform(1, 10) * 365) * 24
        start = get_nearest_holiday_hour(start) - 24
        for i in range (start, start + 24 * 7):
            data.append(proto_function(i))
        plt.plot(data)
        plt.show()

    def show_all():
        data = []
        for i in range (0, 1000000):
            data.append(proto_function(i))
        plt.plot(data)
        plt.show()

    def print_some():
        for i in range(0, 120):
            print(proto_function(i))
        for i in range(1000000, 1000020):
            print(proto_function(i))

    train, dev = initialize_datasets(False)
    start = len(dev) // 2 + 1380
    print(start)
    pdev = dev[start: start + 24 * 7]
    plt.plot(pdev)
    plt.show()

    print_some()
    show_holiday_week()
    show_all()