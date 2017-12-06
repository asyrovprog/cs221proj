import numpy as np
import collections, csv, os
from datetime import datetime, timedelta, date
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

SET_TYPES = ["train", "dev", "test"]
SET_TYPE_SIZES = [0.5, 0.25, 0.25] # proportions of each set

def hoursrange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        for h in range(0, 24):
            yield start_date + timedelta(days=n, hours=h)

def monthdelta(date, delta):
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date.day, [31, 29 if y % 4 == 0 and not y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date.replace(day = d, month = m, year = y)

def preprocess(filename, tofile, fld_date, start_date, end_date, use_filter = False, extended = False):
    FLD_DATE = fld_date
    date_func = hoursrange
    dict_days = collections.defaultdict(int)
    for d in date_func(start_date, end_date):
        dict_days[d] = 0

    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            if len(line[FLD_DATE]) < 10:
                continue
            date = datetime.strptime(line[FLD_DATE], "%m/%d/%Y %I:%M:%S %p")
            date = datetime(date.year, date.month, date.day, date.hour)
            dict_days[date] += 1

    current, weekago, monthago, yearago = [], [], [], []
    for d in date_func(start_date, end_date):
        current.append(dict_days[d])
        weekago.append(dict_days[d - timedelta(days=-7)])
        monthago.append(dict_days[monthdelta(d, -1)])
        yearago.append(dict_days[monthdelta(d, -12)])

    if use_filter:
        current = gaussian_filter(current, sigma=2)
        weekago = gaussian_filter(weekago, sigma=2)
        monthago = gaussian_filter(monthago, sigma=2)
        yearago = gaussian_filter(yearago, sigma=2)

    start = 0
    print("total {} rows".format(len(current)))
    for i in range(0, len(SET_TYPES)):
        file = tofile + "-" + SET_TYPES[i] + ".csv"
        end = start + int(len(current) * SET_TYPE_SIZES[i])
        if extended:
            raw_data = {'current': current[start:end], 'weekago': weekago[start:end], 'monthago': monthago[start:end], 'yearago': yearago[start:end]}
            df = pd.DataFrame(raw_data, columns=["current", "weekago", "monthago", "yearago"])
        else:
            raw_data = {'count': current[start:end]}
            df = pd.DataFrame(raw_data, columns=["count"])
        df.to_csv(file)
        print("created {} with rows from {} to {}".format(file, start, end - 1))
        start = end