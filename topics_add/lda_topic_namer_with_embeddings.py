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
    # Оставляем кортежи (слово, вес) после фильтрации
    filtered = [
        (w, weight) for w, weight in topic_words
        if w.lower() not in stop_words and len(w) > 2
    ]
    filtered.sort(key=lambda x: -x[1])
    return filtered[:top_n]  # Возвращаем кортежи (слово, вес)

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

    # Извлекаем отдельно слова и веса
    words, weights = zip(*filtered_words)  # Распаковываем кортежи

    # Конвертируем веса в тензор
    weights = torch.tensor(weights, dtype=torch.float32)

    # Нормализуем веса
    weights /= torch.sum(weights)

    # Получаем эмбеддинги для слов
    word_embeddings = model.encode(words, convert_to_tensor=True)

    # Вычисляем взвешенное среднее
    topic_embedding = torch.sum(word_embeddings * weights.unsqueeze(1), dim=0)
    # Сравниваем с кандидатами
    cos_scores = util.cos_sim(topic_embedding, candidate_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=3).indices.tolist()
    top_scores = [cos_scores[i].item() for i in top_indices]
    top_names = [candidate_names[i] for i in top_indices]

    combined_name = " / ".join(top_names)
    average_score = round(sum(top_scores) / len(top_scores), 4)

    return combined_name, average_score
