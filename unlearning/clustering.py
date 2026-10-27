from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


class SubclusterDetector:
    def __init__(self, max_clusters: int = 8, sil_threshold: float = 0.15, random_state: int = 0):
        self.max_clusters = max_clusters
        self.sil_threshold = sil_threshold
        self.random_state = random_state
        self.n_clusters_: int = 1
        self.labels_: np.ndarray = np.array([])
        self.silhouette_score_: float = 0.0

    def fit(self, gradient_matrix: np.ndarray) -> "SubclusterDetector":
        n = len(gradient_matrix)
        if n < 4:
            self.labels_ = np.zeros(n, dtype=int)
            return self

        normed = normalize(np.nan_to_num(gradient_matrix, nan=0.0), norm="l2")
        best_k, best_score, best_labels = 1, -1.0, np.zeros(n, dtype=int)

        for k in range(2, min(self.max_clusters + 1, n)):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(normed)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(normed, labels)
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
            if score >= self.sil_threshold:
                break

        if best_k == 1 or best_score < self.sil_threshold:
            self.n_clusters_ = 1
            self.labels_ = np.zeros(n, dtype=int)
            self.silhouette_score_ = 0.0
        else:
            self.n_clusters_ = best_k
            self.labels_ = best_labels
            self.silhouette_score_ = best_score

        return self

    def cluster_indices(self) -> List[List[int]]:
        clusters = []
        for k in range(self.n_clusters_):
            clusters.append(list(np.where(self.labels_ == k)[0]))
        return clusters
