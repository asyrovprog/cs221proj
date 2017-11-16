import numpy as np
import collections, csv, os
from datetime import datetime, timedelta, date
import pandas as pd

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(days=n)

def preprocess_seattle911(filename, tofile):
    FLD_GROUP = "Event Clearance Group"
    FLD_DATETIME = 'Event Clearance Date'
    start_date = datetime(year=2010, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)

    dict_days = collections.defaultdict(int)
    for d in daterange(start_date, end_date):
        dict_days[d] = 0

    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        groups = collections.defaultdict(int)
        group_ids = collections.defaultdict(int)
        for i, line in enumerate(reader):
            if len(line[FLD_DATETIME]) < 2:
                continue
            grp = line[FLD_GROUP]
            if grp not in group_ids:
                group_ids[grp] = len(group_ids)
            grp_id = group_ids[grp]
            groups[line[FLD_GROUP]] += 1
            date = datetime.strptime(line[FLD_DATETIME], "%m/%d/%Y %H:%M:%S %p")
            date = datetime(date.year, date.month, date.day)
            dict_days[date] += 1

    counts = []
    for d in daterange(start_date, end_date):
        counts.append(dict_days[d])

    raw_data = {'count': counts}
    df = pd.DataFrame(raw_data, columns= ["count"])
    df.to_csv(tofile)

if __name__ == "__main__":
    FROMFILE = "../data_raw/seattle-911.csv"
    TOFILE = "../ds/seattle-911-counts-2014-2017.csv"
    print("Creating by day totals for incidents...")
    preprocess_seattle911(FROMFILE, TOFILE)
    print("Saved to " + TOFILE)
