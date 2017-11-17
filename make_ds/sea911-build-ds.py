from utils import *

if __name__ == "__main__":
    FROMFILE = "../data_raw/sea911.csv"
    TOFILE = "../ds/sea911-counts"
    print("Creating by day totals for incidents...")
    start_date = datetime(year=2010, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)
    preprocess(FROMFILE, TOFILE, 'Event Clearance Date', start_date, end_date)
    print("Saved to " + TOFILE)
