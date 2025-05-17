from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache

# Минимальный набор стоп-слов
stop_words = {
    'the', 'and', 'for', 'are', 'any', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'get',
    'who', 'did', 'its',
    'let', 'put', 'too', 'don', 't'
}

# Загружаем модель один раз
model = SentenceTransformer('all-MiniLM-L6-v2')

# Загружаем имена-кандидаты и их векторы
def load_candidate_names(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

candidate_names = load_candidate_names('candidate_names.txt')
candidate_embeddings = model.encode(candidate_names, convert_to_tensor=True)

# Фильтрация и сортировка слов по важности (весу)
def filter_topic_words(topic_words, top_n=20):
    filtered = [
        (w, weight) for w, weight in topic_words
        if w.lower() not in stop_words and len(w) > 2
    ]
    filtered.sort(key=lambda x: -x[1])
    return [w for w, _ in filtered[:top_n]]

# Кэшируемая обёртка
@lru_cache(maxsize=4096)
def get_topic_name_cached(topic_words_str):
    topic_words = eval(topic_words_str)
    return get_topic_name(topic_words)

# Основная функция
def get_topic_name(topic_words, top_n=20):
    filtered_words = filter_topic_words(topic_words, top_n)
    if not filtered_words:
        return "Miscellaneous", 0.0

    # Получаем векторы слов и усредняем
    word_embeddings = model.encode(filtered_words, convert_to_tensor=True)
    topic_embedding = torch.mean(word_embeddings, dim=0, keepdim=True)

    # Сравниваем с кандидатами
    cos_scores = util.cos_sim(topic_embedding, candidate_embeddings)[0]
    best_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_idx].item()

    return candidate_names[best_idx], round(best_score, 4)