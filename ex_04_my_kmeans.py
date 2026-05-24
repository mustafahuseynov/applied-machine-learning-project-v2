from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]


class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.

    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean",
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """

    def __init__(
        self,
        k: int,
        max_iter: int = 100,
        distance_metric: DISTANCE_METRICS = "euclidean",
        init_method: INIT_METHOD = "kmeans++",
    ):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.

        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).

        Returns:
            MyKMeans: Fitted estimator instance.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")

        # Validate input dimensions
        if len(x.shape) not in [2, 3]:
            raise ValueError("Input data must be a 2D or 3D array")

        n_samples = x.shape[0]
        self.centroids = self._initialize_centroids(x)

        for _ in tqdm(range(self.max_iter), desc="Fitting"):
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for i in range(self.k):
                cluster_points = x[labels == i]
                if len(cluster_points) == 0:
                    new_centroids.append(x[np.random.randint(n_samples)])
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))
            new_centroids = np.array(new_centroids)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.inertia_ = np.sum([np.min(distances[i]) for i in range(len(x))])
        self.labels_ = labels  # ← add this line
        return self

    def fit_predict(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.

        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)

        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the specified initialization method.

        Args:
            x (np.ndarray): Training data.

        Returns:
            np.ndarray: Initial centroids.
        """
        n_samples = x.shape[0]

        if self.init_method == "random":
            # Random initialization
            idx = np.random.choice(n_samples, self.k, replace=False)
            return x[idx]

        elif self.init_method == "kmeans++":
            # Choose first centroid randomly
            first_idx = np.random.randint(n_samples)
            centroids = [x[first_idx].copy()]

            # Choose remaining centroids
            for _ in range(1, self.k):
                dists = self._compute_distance(x, np.stack(centroids))
                min_dists = np.min(dists, axis=1)
                probs = min_dists**2
                probs_sum = probs.sum()

                if probs_sum == 0:
                    probs = np.ones(n_samples) / n_samples
                else:
                    probs /= probs_sum

                next_idx = np.random.choice(n_samples, p=probs)
                centroids.append(x[next_idx].copy())

            return np.stack(centroids)

        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance between samples and centroids.

        Args:
            x (np.ndarray): Data points of shape (n_samples, n_features) or (n_samples, time_steps, n_features).
            centroids (np.ndarray): Centroids of shape (k, n_features) or (k, time_steps, n_features).

        Returns:
            np.ndarray: Distances between each sample and each centroid, shape (n_samples, k).
        """
        if self.distance_metric == "euclidean":
            # Compute squared differences, sum over all axes except samples and centroids, then take sqrt
            diff = x[:, None] - centroids[None, :]  # (n_samples, k, ...)
            return np.sqrt(np.sum(diff**2, axis=tuple(range(2, diff.ndim))))

        elif self.distance_metric == "manhattan":
            diff = x[:, None] - centroids[None, :]  # (n_samples, k, ...)
            return np.sum(np.abs(diff), axis=tuple(range(2, diff.ndim)))

        elif self.distance_metric == "dtw":
            return self._dtw(x, centroids)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.

        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)

        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """
        n_samples = x.shape[0]
        k = centroids.shape[0]
        distances = np.zeros((n_samples, k))

        # Handle 2D input (standard feature vectors)
        if x.ndim == 2:
            # For 2D data, treat each sample as a 1D time series
            for i in range(n_samples):
                for j in range(k):
                    distances[i, j] = dtw.distance(x[i], centroids[j])

        elif x.ndim == 3:
            # 3D input: compute mean DTW over features
            for i in range(n_samples):
                for j in range(k):
                    distances[i, j] = np.mean(
                        [
                            dtw.distance(x[i, :, f], centroids[j, :, f])
                            for f in range(x.shape[2])
                        ]
                    )

        else:
            raise ValueError(f"Input data must be 2D or 3D")

        return distances
