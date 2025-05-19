import numpy as np
import json
import pandas as pd
import glob
import os
import html
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import plotly.express as px
from epsfinder import find_optimal_eps

# Конфигурация
EMBEDDINGS_GLOB = 'embeddings_*.npy'
META_FILE = 'lyrics_meta_with_topics.json'
BASE_HTML = 'lyrics_clusters_{name}.html'
COLORS = px.colors.qualitative.Dark24


def process_embeddings(embeddings, meta, filename):
    try:
        # Валидация входных данных
        if len(embeddings) != len(meta):
            raise ValueError(f"Mismatch: embeddings({len(embeddings)}) vs meta({len(meta)})")

        # Кластеризация
        def perform_clustering():
            nonlocal embeddings, filename
            eps_ = find_optimal_eps(embeddings, 4, filename) * 0.9
            dbscan = DBSCAN(eps=eps_, min_samples=4)
            labels = dbscan.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"  Clusters: {n_clusters}, Noise: {n_noise}")
            return labels

        labels = perform_clustering()

        # Проекция t-SNE
        def generate_tsne_projection():
            nonlocal embeddings
            perplexity = min(35, len(embeddings) - 1) if len(embeddings) > 1 else 1
            tsne = TSNE(n_components=2, random_state=10, perplexity=perplexity)
            return tsne.fit_transform(embeddings)

        emb_2d = generate_tsne_projection()

        # Утилиты обработки текста
        def preprocess_text(text, max_lines=10):
            lines = text.split('\n')
            processed = [html.escape(line).replace('\u205f', ' ').replace('\u200b', '') for line in lines]

            short = "<br>".join(processed[:max_lines]) + "..."
            full = "<br>".join(processed)

            if len(lines) > max_lines:
                short += f"""
                <br><button onclick="toggleText('full_{{id}}', 'short_{{id}}', this)" 
                    class="toggle-button">
                    Показать полностью
                </button>
                """
                full += f"""
                <br><button onclick="toggleText('short_{{id}}', 'full_{{id}}', this)" 
                    class="toggle-button">
                    Свернуть
                </button>
                """

            return {'short': short, 'full': full}

        def prepare_topics(items, type_prefix, idx, max_visible=5):
            items = [html.escape(str(item)) for item in items]
            if len(items) <= max_visible:
                return "<br>".join(items)

            visible = items[:max_visible]
            hidden = items[max_visible:]

            return f"""
            <div id="{type_prefix}_short_{idx}">
                {"<br>".join(visible)}
                <button onclick="toggleTopics('{type_prefix}_full_{idx}', '{type_prefix}_short_{idx}', this)" 
                    class="toggle-button">
                    +{len(hidden)} more
                </button>
            </div>
            <div id="{type_prefix}_full_{idx}" style="display:none;">
                {"<br>".join(items)}
                <button onclick="toggleTopics('{type_prefix}_short_{idx}', '{type_prefix}_full_{idx}', this)" 
                    class="toggle-button">
                    - Show less
                </button>
            </div>
            """

        # Подготовка DataFrame
        def prepare_dataframe():
            nonlocal meta, emb_2d, labels
            df_data = []
            for i, m in enumerate(meta):
                text_data = preprocess_text(m['lyrics'])

                df_data.append({
                    'x': emb_2d[i, 0],
                    'y': emb_2d[i, 1],
                    'cluster': labels[i],
                    'artist': m['artist'],
                    'title': m['title'],
                    'lyrics_short': text_data['short'].replace('{id}', f'{i}'),
                    'lyrics_full': text_data['full'].replace('{id}', f'{i}'),
                    'lda_topics': prepare_topics(
                        [t['name'] for t in m.get('lda_topics', [])], 'lda', i),
                    'semantic_topics': prepare_topics(
                        [t['name'] for t in m.get('semantic_topics', [])], 'sem', i),
                    'similarity_topics': prepare_topics(
                        [t['topic'] for t in m.get('topic_word_similarity_scores', [])], 'sim', i),
                    'original_index': i
                })

            df = pd.DataFrame(df_data)
            df['cluster_str'] = df['cluster'].astype(str)
            df.loc[df['cluster'] == -1, 'cluster_str'] = 'noise'
            return df

        df = prepare_dataframe()

        # Визуализация
        def create_plot():
            nonlocal df
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='cluster_str',
                color_discrete_map={c: 'lightgrey' if c == 'noise' else COLORS[i % 24]
                                    for i, c in enumerate(df['cluster_str'].unique())},
                custom_data=['artist', 'title', 'original_index'],
                hover_data={'x': False, 'y': False, 'cluster_str': False}
            )
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
            )
            return fig

        fig = create_plot()

        # Генерация HTML
        def generate_html_content():
            nonlocal fig, meta, df, filename
            tooltip_array = []
            for i, m in enumerate(meta):
                text_data = preprocess_text(m['lyrics'])
                tooltip = f"""
                <div style="padding:10px;">
                    <h3 style="margin:0 0 10px 0; color:#2c3e50;">
                        {m['artist']} - {m['title']}
                    </h3>
                    <div style="margin-bottom:15px; border-bottom:1px solid #eee; padding-bottom:15px;">
                        <div id="short_{i}">{text_data['short'].replace('{id}', f'{i}')}</div>
                        <div id="full_{i}" style="display:none;">{text_data['full'].replace('{id}', f'{i}')}</div>
                    </div>
                    <div style="margin-bottom:15px;">
                        <strong>LDA Topics:</strong><br>
                        {prepare_topics([t['name'] for t in m.get('lda_topics', [])], 'lda', i)}
                        <strong>Semantic Topics:</strong><br>
                        {prepare_topics([t['name'] for t in m.get('semantic_topics', [])], 'sem', i)}
                        <strong>Word Similarity:</strong><br>
                        {prepare_topics([t['topic'] for t in m.get('topic_word_similarity_scores', [])], 'sim', i)}
                    </div>
                </div>
                """.replace("\n", " ")
                tooltip_array.append(tooltip)

            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    .container { display: flex; height: 100vh; }
                    #info-panel { 
                        width: 300px; 
                        padding: 20px; 
                        border-right: 1px solid #ddd; 
                        overflow-y: auto; 
                        font-family: Arial;
                    }
                    #plot { flex-grow: 1; height: 100vh; }
                    .toggle-button {
                        margin-top: 5px;
                        padding: 2px 8px;
                        background: #2c3e50;
                        color: white;
                        border: none;
                        cursor: pointer;
                        border-radius: 3px;
                        font-size: 0.8em;
                    }
                    .clear-button {
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        padding: 10px 20px;
                        background: #2c3e50;
                        color: white;
                        border: none;
                        cursor: pointer;
                    }
                </style>
                <script>
                    let selectedPoint = null;
                    function toggleText(showId, hideId, button) {
                        document.getElementById(showId).style.display = 'block';
                        document.getElementById(hideId).style.display = 'none';
                        button.blur();
                    }
                    function toggleTopics(showId, hideId, button) {
                        document.getElementById(showId).style.display = 'block';
                        document.getElementById(hideId).style.display = 'none';
                        button.blur();
                    }
                    function updateInfo(content) {
                        document.getElementById('info-panel').innerHTML = content;
                    }
                    function clearInfo() {
                        selectedPoint = null;
                        document.getElementById('info-panel').innerHTML = 
                            '<div style="color:#666; padding:20px;">Click a point to view details</div>';
                    }
                    document.addEventListener('DOMContentLoaded', clearInfo);
                </script>
            </head>
            <body>
                <div class="container">
                    <div id="info-panel"></div>
                    <div id="plot"></div>
                </div>
                <button class="clear-button" onclick="clearInfo()">Clear Selection</button>
            """

            html_file = BASE_HTML.format(name=os.path.basename(filename))
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plot'))
                f.write(f"""
                <script>
                    var tooltipArray = {json.dumps(tooltip_array)};
                    document.getElementById('plot').on('plotly_click', function(data) {{
                        if(data.points.length > 0) {{
                            var originalIndex = data.points[0].customdata[2];
                            updateInfo(tooltipArray[originalIndex]);
                        }}
                    }});
                </script>
                """)
            print(f"  Saved: {html_file}")

        generate_html_content()

    except Exception as e:
        print(f"  Error: {str(e)}")


def main():
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_list = list(meta.values())

    for file in glob.glob(EMBEDDINGS_GLOB):
        print(f"\nProcessing: {file}")
        embeddings = np.load(file)
        process_embeddings(embeddings, meta_list[:len(embeddings)], file)


if __name__ == "__main__":
    main()