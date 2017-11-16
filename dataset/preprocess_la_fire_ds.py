import numpy as np
import collections, csv, os
from datetime import datetime, timedelta, date
import pandas as pd

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(days=n)

def preprocess_la_fire(filename, tofile):
    FLD_DATE = 'Received DtTm'
    start_date = datetime(year=2001, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)

    dict_days = collections.defaultdict(int)
    for d in daterange(start_date, end_date):
        dict_days[d] = 0

    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        groups = collections.defaultdict(int)
        group_ids = collections.defaultdict(int)
        for i, line in enumerate(reader):
            date = datetime.strptime(line[FLD_DATE], "%m/%d/%Y %H:%M:%S %p")
            date = datetime(date.year, date.month, date.day)
            dict_days[date] += 1

    counts = []
    for d in daterange(start_date, end_date):
        counts.append(dict_days[d])

    raw_data = {'count': counts}
    df = pd.DataFrame(raw_data, columns= ["count"])
    df.to_csv(tofile)

if __name__ == "__main__":
    FROMFILE = "../data_raw/los-angeles-fire.csv"
    TOFILE = "../ds/los-angeles-fire-counts-2001-2017.csv"
    print("Creating by day totals for incidents...")
    preprocess_la_fire(FROMFILE, TOFILE)
    print("Saved to " + TOFILE)
