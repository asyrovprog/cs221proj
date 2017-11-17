import numpy as np
import collections, csv, os
from datetime import datetime, timedelta, date
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

SET_TYPES = ["train", "dev", "test"]
SET_TYPE_SIZES = [0.5, 0.25, 0.25] # proportions of each set


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(days=n)


def hoursrange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        for h in range(0, 24):
            yield start_date + timedelta(days=n, hours=h)


def preprocess(filename, tofile, fld_date, start_date, end_date, days = True, use_filter = False):
    FLD_DATE = fld_date
    date_func = daterange if days else hoursrange
    dict_days = collections.defaultdict(int)
    for d in date_func(start_date, end_date):
        dict_days[d] = 0

    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            if len(line[FLD_DATE]) < 10:
                continue
            date = datetime.strptime(line[FLD_DATE], "%m/%d/%Y %I:%M:%S %p")
            if days:
                date = datetime(date.year, date.month, date.day)
            else:
                date = datetime(date.year, date.month, date.day, date.hour)
            dict_days[date] += 1

    counts = []
    for d in date_func(start_date, end_date):
        counts.append(dict_days[d])

    if use_filter:
        counts = gaussian_filter(counts, sigma=2)

    start = 0
    print("total {} rows".format(len(counts)))
    for i in range(0, len(SET_TYPES)):
        file = tofile + "-" + SET_TYPES[i] + ".csv"
        end = start + int(len(counts) * SET_TYPE_SIZES[i])
        raw_data = {'count': counts[start:end]}
        df = pd.DataFrame(raw_data, columns=["count"])
        df.to_csv(file)
        print("created {} with rows from {} to {}".format(file, start, end - 1))
        start = end