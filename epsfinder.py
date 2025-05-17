import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Путь к эмбеддингам и метаданным — подкорректируй, если надо
EMBEDDINGS_PATH = 'lyrics_with_topics_embeddings.npy'
META_PATH = 'lyrics_with_topics_meta_embeddings.json'


def load_embeddings(path):
    return np.load(path)


def plot_k_distance(embeddings, k=4):
    # k = min_samples, рекомендую 4 (то есть 3 соседей + сама точка)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(embeddings)
    distances, indices = neighbors_fit.kneighbors(embeddings)

    # Берём расстояния до k-го соседа
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Точки, отсортированные по расстоянию до {}-го соседа'.format(k))
    plt.ylabel('Расстояние')
    plt.title('График выбора параметра eps для DBSCAN (k-distance plot)')
    plt.grid(True)
    plt.show()


def main():
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"Загружено эмбеддингов: {embeddings.shape[0]} песен, размерность {embeddings.shape[1]}")

    # Здесь указывай min_samples, например 4
    min_samples = 4
    plot_k_distance(embeddings, k=min_samples)


if __name__ == '__main__':
    main()
