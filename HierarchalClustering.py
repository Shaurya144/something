# Hierarchical clustering is an unsupervised learning method for clustering data points. 
# Unsupervised means it does not need to be trained
# Hierarchical clustering requires us to decide on both a distance and linkage method.
# Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up approach. like a dendrogram

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean') # we use ward and euclidean as they attempt to minimize the variance between clusters
dendrogram(linkage_data)

plt.show()