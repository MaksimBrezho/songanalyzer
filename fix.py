import json

with open("songs_translated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_data = {}
swapped_keys = []

for key, entry in data.items():
    if entry.get("no_lyrics") is True and "original_artist" in entry:
        # Меняем местами title и artist
        entry["title"], entry["artist"] = entry.get("artist", ""), entry.get("title", "")
        swapped_keys.append(key)
    fixed_data[key] = entry

# Сохраняем исправленные данные
with open("songs_translated.json", "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=4)

# Логируем, какие ключи были исправлены
print(f"✅ Обмен названия и артиста выполнен для {len(swapped_keys)} записей:")
for k in swapped_keys:
    print(f" - {k}")
