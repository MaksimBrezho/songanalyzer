import json

def simplify_lyrics_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    simplified_data = {}

    for key, info in data.items():
        # Пропускаем, если явно указано отсутствие текста
        if info.get('no_lyrics', False):
            continue

        artist = info.get('artist') or info.get('original_artist')
        title = info.get('title') or info.get('original_title')
        lyrics = info.get('lyrics', '')
        lyrics_en = info.get('lyrics_en', '')

        # Пропускаем, если и оригинального, и текущего исполнителя нет
        if not artist or not title:
            continue

        simplified_data[key] = {
            'artist': artist,
            'title': title,
            'lyrics': lyrics,
            'lyrics_en': lyrics_en
        }

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_data, outfile, ensure_ascii=False, indent=4)

# Пример использования
simplify_lyrics_json('songs_translated2.json', 'simplified_lyrics.json')
