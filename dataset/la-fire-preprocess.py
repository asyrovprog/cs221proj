from dataset.utils import *

if __name__ == "__main__":
    FROMFILE = "../data_raw/los-angeles-fire.csv"
    TOFILE = "../ds/los-angeles-fire-counts-2001-2017.csv"
    print("Creating by day totals for incidents...")
    start_date = datetime(year=2001, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date)
    print("Saved to " + TOFILE)

    TOFILE = "../ds/los-angeles-fire-counts-2001-2017-hours.csv"
    print("Creating by day totals for incidents...")
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date, False)
    print("Saved to " + TOFILE)
