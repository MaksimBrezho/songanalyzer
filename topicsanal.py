import json
from collections import defaultdict

# Загрузка данных (замените на ваш способ загрузки)
with open('lyrics_meta_with_topics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Инициализация счетчиков для каждой группы
lda_stats = defaultdict(int)
semantic_stats = defaultdict(int)
word_similarity_stats = defaultdict(int)

# Обработка данных
for song in data.values():
    # LDA topics
    for topic in song['lda_topics']:
        if topic['prob'] > 0.0:
            lda_stats[topic['name']] += 1

    # Semantic topics
    for topic in song['semantic_topics']:
        if topic['similarity'] > 0.0:
            semantic_stats[topic['name']] += 1

    # Word similarity
    for topic in song['topic_word_similarity_scores']:
        if topic['score'] > 0.0:
            word_similarity_stats[topic['topic']] += 1


# Функция для вывода статистики
def print_stats(stats_dict, group_name):
    print(f"\n{group_name} (топики > 0):")
    if not stats_dict:
        print("  Нет данных")
        return
    sorted_stats = sorted(stats_dict.items(), key=lambda x: (-x[1], x[0]))
    for topic, count in sorted_stats:
        print(f"  {topic}: {count}")


# Вывод результатов
print_stats(lda_stats, "LDA-топики")
print_stats(semantic_stats, "Семантические топики")
print_stats(word_similarity_stats, "Схожесть слов")
