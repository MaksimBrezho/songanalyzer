import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Загрузка данных
with open('lyrics_meta_with_topics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

combined_embeddings = []
meta = []

print("Обработка песен и генерация эмбеддингов...")

for key in tqdm(data, desc="Песни"):
    val = data[key]
    lyrics_en = val['lyrics_en']
    lda_topics = val.get('lda_topics', [])

    # Лирика
    lyrics_embedding = model.encode(lyrics_en)

    # Темы
    if lda_topics:
        topic_texts = [t['name'] for t in lda_topics]
        topic_weights = np.array([t['prob'] for t in lda_topics])
        topic_embeds = model.encode(topic_texts)
        topic_embedding = np.average(topic_embeds, axis=0, weights=topic_weights)
    else:
        topic_embedding = np.zeros(model.get_sentence_embedding_dimension())

    # Объединение
    combined_embedding = 0.51 * lyrics_embedding + 0.49 * topic_embedding
    combined_embeddings.append(combined_embedding)

    meta.append({
        'artist': val['artist'],
        'title': val['title'],
        'lyrics': val['lyrics'],
        'lyrics_en': lyrics_en,
        'lda_topics': lda_topics
    })

# Сохранение результатов
np.save('lyrics_with_topics_embeddings.npy', np.array(combined_embeddings))

with open('lyrics_with_topics_meta_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Готово!")
