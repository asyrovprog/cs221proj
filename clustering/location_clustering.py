from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from pandas import read_csv
from PIL import Image
import argparse

from dataset_descriptor import SeattlePoliceDataset
from dataset_descriptor import SanFranciscoFireDataset

parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', default=10, type=int)
args = parser.parse_args()
n_clusters = args.n

np.random.seed(42)
dataset = SeattlePoliceDataset()
#dataset = SanFranciscoFireDataset()

reduced_data = dataset.getLocationsData()
kmeans = KMeans(init='k-means++',
                n_clusters=n_clusters,
                n_init=n_clusters)
kmeans.fit(reduced_data)

# Step size of the mesh.
h = .0001     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = dataset.getXBoundaries(reduced_data)
y_min, y_max = dataset.getYBoundaries(reduced_data)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#print(x_min, x_max, y_min, y_max) 

# Obtain labels for each point in mesh.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# Plot San Francisco image
map_img = dataset.getBackgroundImage()
plt.imshow(map_img,
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           alpha = 0.5)

# Plot the centroids as red X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=1,
            color='r', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
