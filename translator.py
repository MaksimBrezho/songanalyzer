import json
import logging
from langdetect import detect as detect_langdetect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import langid
from deep_translator import GoogleTranslator
import os

DetectorFactory.seed = 0

logging.basicConfig(
    filename='translation_errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

INPUT_FILE = 'processed_data_clear.json'
OUTPUT_FILE = 'songs_translated2.json'
SUSPICIOUS_LOG = 'suspicious_songs.log'

def get_langdetect_info(text):
    try:
        lang_detect = detect_langdetect(text)
        langs = detect_langs(text)
        return lang_detect, langs
    except LangDetectException:
        return "unknown", []

def detect_full_text_lang(text):
    langid_lang, _ = langid.classify(text)
    langdetect_lang, langdetect_probs = get_langdetect_info(text)
    top_lang = langdetect_probs[0].lang if langdetect_probs else "unknown"
    top_prob = langdetect_probs[0].prob if langdetect_probs else 0.0
    return langid_lang, langdetect_lang, langdetect_probs, top_lang, top_prob

def translate_to_en(text, src_lang):
    if src_lang in ("en", "unknown"):
        return text
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except Exception as e:
        logging.warning(f"Translation failed for lang={src_lang}: {text[:50]}... Error: {e}")
        return None

def process_song(key, song):
    lyrics = song.get("lyrics", "")
    if not lyrics.strip():
        return {"lang": "unknown", "lyrics_en": ""}

    lines = lyrics.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    total_non_empty = len(non_empty_lines)

    langid_lang, langdetect_lang, langdetect_probs, top_lang, top_prob = detect_full_text_lang(lyrics)

    if langid_lang == langdetect_lang == top_lang and top_prob >= 0.9:
        translated = translate_to_en(lyrics, top_lang)
        return {
            "lang": top_lang,
            "lyrics_en": translated if translated else "",
            "langid_lang": langid_lang,
            "langdetect_lang": langdetect_lang,
            "lang_probs": [f"{lp.lang}:{lp.prob:.2f}" for lp in langdetect_probs]
        }

    line_lang_map = {}
    lang_lines = {}

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        lid_lang, _ = langid.classify(line)
        try:
            ldet_lang = detect_langdetect(line)
        except LangDetectException:
            ldet_lang = "unknown"

        if lid_lang == ldet_lang:
            lang_lines.setdefault(lid_lang, []).append((idx, line))
            line_lang_map[idx] = lid_lang
        else:
            line_lang_map[idx] = "unknown"

    lang_keys = list(lang_lines.keys())
    full_lang_label = "-".join(sorted(set(lang_keys))) if lang_keys else "unknown"

    translated_lines = lines.copy()
    confident = True

    for lang_code, items in lang_lines.items():
        group_text = "\n".join([line for _, line in items])
        lid, ldet, lprobs, top, prob = detect_full_text_lang(group_text)
        if lid == ldet == top and prob >= 0.9:
            translated_text = translate_to_en(group_text, top)
            if translated_text:
                translated_parts = translated_text.splitlines()
                for (idx, _), t_line in zip(items, translated_parts):
                    translated_lines[idx] = t_line
        else:
            confident = False
            break

    if not confident:
        translated = translate_to_en(lyrics, top_lang)
        with open(SUSPICIOUS_LOG, 'a', encoding='utf-8') as logf:
            logf.write(f"{key}\n")
        return {
            "lang": top_lang,
            "lyrics_en": translated if translated else "",
            "langid_lang": langid_lang,
            "langdetect_lang": langdetect_lang,
            "lang_probs": [f"{lp.lang}:{lp.prob:.2f}" for lp in langdetect_probs],
            "lang_per_line": line_lang_map
        }

    return {
        "lang": full_lang_label,
        "lyrics_en": "\n".join(translated_lines),
        "langid_lang": langid_lang,
        "langdetect_lang": langdetect_lang,
        "lang_probs": [f"{lp.lang}:{lp.prob:.2f}" for lp in langdetect_probs],
        "lang_per_line": line_lang_map
    }

# Загрузка исходных данных
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Если файл уже существует, подгружаем его, чтобы не перезаписывать
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
else:
    translated_data = {}

# Обработка песен
for key, song in data.items():
    if key in translated_data:
        continue

    result = process_song(key, song)
    translated_data[key] = {
        **song,
        **result
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Обработано: {key} — язык: {result['lang']}")

print("🎉 Завершено. Все переводы сохранены.")
