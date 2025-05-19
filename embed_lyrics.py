import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Загрузка данных
print("🔄 Загрузка данных...")
with open('lyrics_meta_with_topics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('topics_info.json', 'r', encoding='utf-8') as f:
    topics_info = json.load(f)
with open('theme_to_words.json', 'r', encoding='utf-8') as f:
    theme_to_words = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
dim = model.get_sentence_embedding_dimension()

# Преобразование topics_info в словарь
print("⚙️ Подготовка данных...")
topics_info_dict = {}
for t in tqdm(topics_info, desc="Обработка topics_info"):
    topics_info_dict[t['topic_id']] = t

# Обработка каждой песни
all_embeddings = []
print("\n🎵 Обработка песен:")
song_iter = tqdm(data.items(), total=len(data), desc="Песни")

for key, song in song_iter:
    embeddings = {
        'a': np.zeros(dim),
        'b1': np.zeros(dim),
        'b2': np.zeros(dim),
        'c1': np.zeros(dim),
        'c2': np.zeros(dim),
        'd1': np.zeros(dim),
        'd2': np.zeros(dim)
    }

    # Обработка lyrics_en
    lyrics_en = song.get('lyrics_en', '')
    if lyrics_en:
        embeddings['a'] = model.encode(lyrics_en)

    # Обработка LDA topics
    lda_iter = tqdm(song.get('lda_topics', []), desc="LDA темы", leave=False)
    for topic in lda_iter:
        if topic.get('prob', 0) <= 0:
            continue

        # LDA names
        if name := topic.get('name', ''):
            embeddings['b1'] += model.encode(name) * topic['prob']

        # LDA words
        if topic_id := topic.get('topic'):
            if words := topics_info_dict.get(topic_id, {}).get('words', []):
                word_embs = model.encode(words)
                embeddings['b2'] += np.mean(word_embs, axis=0) * topic['prob']

    # Обработка semantic topics
    semantic_iter = tqdm(song.get('semantic_topics', []), desc="Семантические темы", leave=False)
    for topic in semantic_iter:
        if topic.get('similarity', 0) <= 0:
            continue

        # Semantic names
        if name := topic.get('name', ''):
            embeddings['c1'] += model.encode(name) * topic['similarity']

        # Semantic words
        if topic_id := topic.get('topic'):
            if words := topics_info_dict.get(topic_id, {}).get('words', []):
                word_embs = model.encode(words)
                embeddings['c2'] += np.mean(word_embs, axis=0) * topic['similarity']

    # Обработка topic_word_similarity
    twss_iter = tqdm(song.get('topic_word_similarity_scores', []), desc="TWSS", leave=False)
    for entry in twss_iter:
        if entry.get('score', 0) <= 0 or not (topic_name := entry.get('topic', '')):
            continue

        # TWSS names
        embeddings['d1'] += model.encode(topic_name) * entry['score']

        # TWSS words
        if word_entries := theme_to_words.get(topic_name, []):
            words = [we[0] for we in word_entries if we[1] > 0]
            if words:
                word_embs = model.encode(words)
                embeddings['d2'] += np.mean(word_embs, axis=0) * entry['score']

    all_embeddings.append(embeddings)

# Сохранение компонентов
print("\n💾 Сохранение компонентов:")
component_names = ['a', 'b1', 'b2', 'c1', 'c2', 'd1', 'd2']
for name in tqdm(component_names, desc="Компоненты"):
    embs = np.array([e[name] for e in all_embeddings])
    np.save(f'embeddings_{name}.npy', embs)

# Комбинации
print("\n🧩 Комбинирование эмбеддингов:")
weights = {
    'lda': (0.5, 0.5),
    'semantic': (0.5, 0.5),
    'twss': (0.5, 0.5)
}


# Загрузка компонентов с прогресс-баром
def load_with_progress(files):
    return [np.load(f) for f in tqdm(files, desc="Загрузка данных")]


print("\n🔗 Комбинации внутри категорий:")
b1, b2 = load_with_progress(['embeddings_b1.npy', 'embeddings_b2.npy'])
lda_combined = weights['lda'][0] * b1 + weights['lda'][1] * b2
np.save('embeddings_lda_combined.npy', lda_combined)

c1, c2 = load_with_progress(['embeddings_c1.npy', 'embeddings_c2.npy'])
semantic_combined = weights['semantic'][0] * c1 + weights['semantic'][1] * c2
np.save('embeddings_semantic_combined.npy', semantic_combined)

d1, d2 = load_with_progress(['embeddings_d1.npy', 'embeddings_d2.npy'])
twss_combined = weights['twss'][0] * d1 + weights['twss'][1] * d2
np.save('embeddings_twss_combined.npy', twss_combined)

# Комбинации с текстом
print("\n📝 Комбинации с текстом:")
a = np.load('embeddings_a.npy')

combinations = [
    ('lyrics_plus_lda', (0.7, 0.3), [a, lda_combined]),
    ('lyrics_plus_semantic', (0.7, 0.3), [a, semantic_combined]),
    ('lyrics_plus_twss', (0.7, 0.3), [a, twss_combined])
]

for name, w, data in tqdm(combinations, desc="Текст+топики"):
    combined = w[0] * data[0] + w[1] * data[1]
    np.save(f'embeddings_{name}.npy', combined)

# Полные комбинации
print("\n🌈 Полные комбинации:")
all_weights = {
    'all_components_equal': (0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14),
    'text_heavy': (0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05)
}

components = [a, b1, b2, c1, c2, d1, d2]
for combo_name, w in tqdm(all_weights.items(), desc="Комбинации"):
    combined = sum(wi * c for wi, c in zip(w, components))
    np.save(f'embeddings_{combo_name}.npy', combined)

print("\n✅ Готово! Все эмбеддинги сохранены!")