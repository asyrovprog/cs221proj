from dataset.utils import *

if __name__ == "__main__":
    FROMFILE = "../data_raw/seattle-911.csv"
    TOFILE = "../ds/seattle-911-counts-2014-2017.csv"
    print("Creating by day totals for incidents...")
    start_date = datetime(year=2010, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)
    preprocess(FROMFILE, TOFILE, 'Event Clearance Date', start_date, end_date)
    print("Saved to " + TOFILE)
