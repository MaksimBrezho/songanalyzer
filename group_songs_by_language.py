import json
from collections import defaultdict

INPUT_FILE = 'songs_translated2.json'
OUTPUT_FILE = 'songs_by_language.json'

# Загружаем файл с переводами
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Группировка: язык → автор → список песен
grouped = defaultdict(lambda: defaultdict(list))

for key, song in data.items():
    lang = song.get("lang", "unknown")
    artist = song.get("artist", "Unknown Artist")
    title = song.get("title", "Untitled")

    grouped[lang][artist].append(title)

# Преобразуем defaultdict в обычный dict
grouped = {lang: dict(artists) for lang, artists in grouped.items()}

# Сохраняем сгруппированный файл
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(grouped, f, ensure_ascii=False, indent=4)

# Печатаем статистику
print("📊 Статистика по языкам:")
for lang, artists in sorted(grouped.items(), key=lambda x: -sum(len(songs) for songs in x[1].values())):
    song_count = sum(len(songs) for songs in artists.values())
    print(f"{lang}: {song_count} песен, {len(artists)} авторов")

print(f"\n✅ Группировка завершена. Сохранено в '{OUTPUT_FILE}'")
