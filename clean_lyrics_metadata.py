import json
import sys
import os

def clean_text(text):
    # Заменяем \u2005 (FOUR-PER-EM SPACE) на пробел и удаляем \u200b (ZERO WIDTH SPACE)
    return text.replace('\u2005', ' ').replace('\u200b', '')

def clean_json(data):
    cleaned = {}
    for key, value in data.items():
        new_entry = {}
        for k, v in value.items():
            if isinstance(v, str):
                new_entry[k] = clean_text(v)
            else:
                new_entry[k] = v
        cleaned[clean_text(key)] = new_entry
    return cleaned

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'processed_data.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'processed_data_clear.json'

    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = clean_json(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    print(f"Обработка завершена. Результат сохранён в {output_path}")

if __name__ == '__main__':
    main()
