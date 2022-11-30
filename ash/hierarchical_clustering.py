import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy


def example():
    dataset = pd.read_csv("D:\Study\diplomka2\\ash\\ash\Mall_Customers.csv")

    X = dataset.iloc[:, [3, 4]].values
    Z = hierarchy.linkage(X, "single")

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method="ward"))

    model = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
    model.fit(X)
    labels = model.labels_

    # plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
    # plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
    # plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
    # plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
    # plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
    # plt.show()

    return X, labels, Z, dendrogram
