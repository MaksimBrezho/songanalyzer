import json
import torch
from sentence_transformers import SentenceTransformer, util
from gensim.corpora import Dictionary

# Загрузка модели и словаря
model = SentenceTransformer('all-MiniLM-L6-v2')
dictionary = Dictionary.load('lyrics.dict')

# Загрузка тем
with open('candidate_names.txt', encoding='utf-8') as f:
    themes = [line.strip() for line in f if line.strip()]

# Все слова из словаря
all_words = [dictionary[id] for id in dictionary.keys()]
word_embeddings = model.encode(all_words, convert_to_tensor=True)

# Подбор слов к темам
theme_to_top_words = {}
for theme in themes:
    theme_emb = model.encode(theme, convert_to_tensor=True)
    cos_sim = util.cos_sim(theme_emb, word_embeddings)[0]
    top_k = torch.topk(cos_sim, 20)

    top_words = [all_words[i] for i in top_k.indices]
    top_scores = [cos_sim[i].item() for i in top_k.indices]

    theme_to_top_words[theme] = list(zip(top_words, top_scores))

# Сохранение результата
with open('theme_word_matches.json', 'w', encoding='utf-8') as f:
    json.dump(theme_to_top_words, f, ensure_ascii=False, indent=2)
