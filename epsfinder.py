import numpy as np
import json
import glob
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Конфигурация
EMBEDDINGS_GLOB = 'embeddings_*.npy'
META_PATH = 'lyrics_meta_with_topics.json'
PLOTS_DIR = 'eps_plots'
K = 4  # min_samples для DBSCAN


def find_optimal_eps(embeddings, k, filename):
    # Вычисление k-расстояний
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)
    k_distances = np.sort(distances[:, k - 1])

    # Поиск точки изгиба
    kneedle = KneeLocator(
        x=range(len(k_distances)),
        y=k_distances,
        curve='convex',
        direction='increasing'
    )

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, label='K-distance curve')

    if kneedle.elbow is not None:
        eps = k_distances[kneedle.elbow]
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--',
                    label=f'Estimated eps: {eps:.2f}')
    else:
        eps = None

    plt.title(f'K-distance plot for {os.path.basename(filename)}\n(k={K})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {K}-th neighbor')
    plt.legend()
    plt.grid(True)

    # Сохранение графика
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f'knee_{os.path.basename(filename)}.png')
    plt.savefig(plot_path)
    plt.close()

    return eps


def process_embeddings(filename):
    try:
        embeddings = np.load(filename)
        print(f"\nProcessing: {filename}")
        print(f"Embeddings shape: {embeddings.shape}")

        eps = find_optimal_eps(embeddings, K, filename)

        if eps is not None:
            print(f"Recommended eps: {eps:.4f}")
            return {
                'filename': filename,
                'eps': eps,
                'min_samples': K
            }
        else:
            print("Warning: No clear elbow detected!")
            return None

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def main():
    # Создаем директорию для графиков
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Находим все файлы эмбеддингов
    embed_files = glob.glob(EMBEDDINGS_GLOB)

    if not embed_files:
        print("No embedding files found!")
        return

    results = []

    for file in embed_files:
        result = process_embeddings(file)
        if result:
            results.append(result)

    # Сохраняем результаты
    if results:
        with open('eps_recommendations.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nRecommendations saved to eps_recommendations.json")


if __name__ == '__main__':
    main()