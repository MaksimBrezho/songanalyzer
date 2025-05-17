import json
import re
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

def preprocess_lyrics(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s']", ' ', text)
    words = word_tokenize(text, language='english')
    lemmatized = []
    for w in words:
        try:
            lemma = lemmatizer.lemmatize(w)
            lemmatized.append(lemma)
        except Exception:
            lemmatized.append(w)
    # Убираем повторы, сохраняя порядок
    seen = set()
    unique_words = []
    for w in lemmatized:
        if w not in seen:
            unique_words.append(w)
            seen.add(w)
    return ' '.join(unique_words)


# Читаем исходный JSON
with open('simplified_lyrics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Обрабатываем каждый элемент
processed_data = {}
for key, song in data.items():
    lyrics_en = song.get('lyrics_en', '')
    processed_lyrics = preprocess_lyrics(lyrics_en)
    # Копируем весь словарь и заменяем lyrics_en
    new_song = dict(song)
    new_song['lyrics_en'] = processed_lyrics
    processed_data[key] = new_song

# Сохраняем в новый JSON
with open('simplified_lyrics_processed.json', 'w', encoding='utf-8') as f_out:
    json.dump(processed_data, f_out, ensure_ascii=False, indent=2)

print("Предобработка завершена. Результат сохранён в simplified_lyrics_processed.json")
