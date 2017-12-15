import numpy as np
from pandas import read_csv
from PIL import Image
from ast import literal_eval as make_tuple

class SeattlePoliceDataset(object):
    def getLocationsData(self):
        raw_data = read_csv('1week_may2017.csv').as_matrix()
        locations_data = raw_data[:,12:14]
        return locations_data

    def getClustersCount(self):
        return 10

    def getXBoundaries(self, data):
        return (-122.4530924, -122.2042264)

    def getYBoundaries(self, data):
        return (47.4682218, 47.76240258)

    def getBackgroundImage(self):
        return Image.open("seattle_map.png")

class SanFranciscoFireDataset(object):
    def getLocationsData(self):
        data = read_csv('sf_fire_1week_june2004.csv').as_matrix()
        l = []
        for row in data:
            t = make_tuple(row[32]);
            l.append([t[1],t[0]]);
        return np.array(l)

    def getClustersCount(self):
        return 10

    def getXBoundaries(self, data):
        return (-122.5431111, -122.3368721)

    def getYBoundaries(self, data):
        return (37.69260353, 37.84191253)

    def getBackgroundImage(self):
        return Image.open("sf_map.png")
