import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

if __name__ == '__main__':

    data =pd.read_csv("../credit_card_data.csv")
    missing = data.isna().sum()
    print(missing)

    data = data.fillna(data.median())
    print(data.isna().sum())

    # Let's assume we use all cols except CustomerID
    vals = data.iloc[:, 1:].values

    # Use the Elbow method to find a good number of clusters using WCSS
    wcss = []
    for ii in range(1, 30):
        kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit_predict(vals)
        wcss.append(kmeans.inertia_)

    plt.plot(wcss, 'ro-', label="WCSS")
    plt.title("Computing WCSS for KMeans++")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

    best_cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", "PURCHASES_TRX"]
    # iz lakta vidimo da je optimalan broj klastera oko 6
    kmeans = KMeans(n_clusters=6, init="k-means++", n_init=10, max_iter=300)
    best_vals = data[best_cols].iloc[:, 1:].values
    y_pred = kmeans.fit_predict(best_vals)
    data["cluster"] = y_pred
    best_cols.append("cluster")

    sns.pairplot(data[best_cols], hue="cluster")
    plt.show()

