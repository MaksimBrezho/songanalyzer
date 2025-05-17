import numpy as np
import json
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import plotly.express as px

# Загрузка эмбеддингов
embeddings = np.load('lyrics_with_topics_embeddings.npy')

# Загрузка метаданных
with open('lyrics_with_topics_meta_embeddings.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Кластеризация DBSCAN
dbscan = DBSCAN(eps=0.52, min_samples=4)
labels = dbscan.fit_predict(embeddings)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Количество найденных кластеров (без шума): {n_clusters}")
print(f"Количество песен, отнесённых к шуму: {n_noise}")

# Проекция в 2D с помощью t-SNE
n_samples = embeddings.shape[0]
perplexity = min(35, n_samples - 1) if n_samples > 1 else 1
tsne = TSNE(n_components=2, random_state=10, perplexity=perplexity)
emb_2d = tsne.fit_transform(embeddings)

# Подготовка html-текста с <br> для переносов
lyrics_list = [m['lyrics'].replace('\n', '<br>') for m in meta]

# Формируем DataFrame
df = pd.DataFrame({
    'x': emb_2d[:, 0],
    'y': emb_2d[:, 1],
    'cluster': labels,
    'artist': [m['artist'] for m in meta],
    'title': [m['title'] for m in meta],
    'lyrics_html': lyrics_list
})

# Преобразуем метки в строку, шум обозначим "noise"
df['cluster_str'] = df['cluster'].astype(str)
df.loc[df['cluster'] == -1, 'cluster_str'] = 'noise'

# Дискретная палитра Plotly (24 ярких цвета)
colors = px.colors.qualitative.Dark24

# Создаём цветовую последовательность с серым для шума
unique_clusters = df['cluster_str'].unique().tolist()
color_map = {}
for i, cluster in enumerate(sorted(unique_clusters)):
    if cluster == 'noise':
        color_map[cluster] = 'lightgrey'
    else:
        color_map[cluster] = colors[i % len(colors)]

fig = px.scatter(
    df,
    x='x',
    y='y',
    color='cluster_str',
    color_discrete_map=color_map,
    hover_data=['artist', 'title'],
    title='Кластеризация песен (DBSCAN + t-SNE)'
)

fig.update_layout(
    legend_title_text='Кластер',
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

html_filename = 'lyrics_clusters_dbscan_colored.html'
fig.write_html(html_filename)
print(f"График сохранён в {html_filename}. Откройте этот файл в браузере для просмотра.")
