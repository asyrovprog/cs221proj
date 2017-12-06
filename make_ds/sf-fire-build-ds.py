from utils import *

if __name__ == "__main__":
    FROMFILE = "../data_raw/sf-fire.csv"
    start_date = datetime(year=2001, month=1, day=1)
    end_date = datetime(year=2017, month=10, day=1)

    TOFILE = "../ds/sf-fire-simple"
    print("Creating by day totals for incidents...")
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date)
    print("Saved to " + TOFILE + "...")

    TOFILE = "../ds/sf-fire-simple-gauss"
    print("Creating by day totals for incidents with gaussian filter...")
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date, False, True)
    print("Saved to " + TOFILE)

    TOFILE = "../ds/sf-fire-extended"
    print("Creating by day totals for incidents...")
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date, False, True)
    print("Saved to " + TOFILE + "...")

    TOFILE = "../ds/sf-fire-extended-gauss"
    print("Creating by day totals for incidents with gaussian filter...")
    preprocess(FROMFILE, TOFILE, 'Received DtTm', start_date, end_date, True, True)
    print("Saved to " + TOFILE)