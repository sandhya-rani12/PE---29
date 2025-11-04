import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def elbow_method(X_scaled, save_path="../outputs/elbow.png"):
    sse = []
    K = range(1, 11)
    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        sse.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(K, sse, marker='o')
    plt.title("Elbow Method - Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("SSE (Inertia)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train_kmeans(X_scaled, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels


def plot_clusters(df, model, scaler, save_path="../outputs/clusters.png"):
    centroids = scaler.inverse_transform(model.cluster_centers_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='tab10',
        s=70,
        alpha=0.8
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        c='black',
        marker='X',
        label='Centroids'
    )
    plt.title("Customer Clusters")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
