from tempfile import mktemp
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import os

datasets = [
    # This incidents Response Seattle-Police (some missing data in 2013-2014)
    ["https://data.seattle.gov/api/views/3k2p-39jp/rows.csv?accessType=DOWNLOAD",
     False,
     "./data_raw/sea911.csv"],
    # Individual household electric power consumption Data Set
    # 2075259 measurements gathered between December 2006 and November 2010 (47 months).
    # ["https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
    #  True,
    # "./data/", "./data_raw/household_power_consumption.zip"],
    # The City of San Francisco Fire Department Calls for Service
    # https://catalog.data.gov/dataset/fire-department-calls-for-service
    ["https://data.sfgov.org/api/views/nuek-vuh3/rows.csv?accessType=DOWNLOAD",
     False,
    "./data_raw/sf-fire.csv"]]


def download_unzip(download_info):
    if download_info[1]:
        if not os.path.isfile(download_info[3]):
            urllib.request.urlretrieve(download_info[0], download_info[3])
        fl = ZipFile(download_info[3])
        fl.extractall(download_info[2])
        fl.close()
    else:
        if not os.path.isfile(download_info[2]):
            urllib.request.urlretrieve(download_info[0], download_info[2])
        else:
            print("file already downloaded: " + di[2])


if __name__ == "__main__":
    for di in datasets:
        if not os.path.isdir("./data_raw"):
            os.mkdir("./data_raw")
        print("downloading " + di[0] + "...")
        download_unzip(di)
        print("downloaded to: " + di[2])

