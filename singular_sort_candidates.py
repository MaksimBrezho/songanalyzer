import inflect

# Инициализируем конвертер для работы с числами и словами
p = inflect.engine()

# Файл с исходными словами
input_file = "candidate_names.txt"
output_file = "candidate_names_singular_sorted.txt"

def singularize_phrase(phrase):
    words = phrase.strip().split()
    # Преобразуем только последнее слово к единственному числу
    if words:
        words[-1] = p.singular_noun(words[-1]) or words[-1]
    return ' '.join(words)

# Чтение, обработка и сортировка
with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Приведение к единственному числу
singular_lines = [singularize_phrase(line) for line in lines]

# Сортировка по алфавиту
sorted_singular_lines = sorted(singular_lines, key=lambda x: x.lower())

# Запись результата
with open(output_file, "w", encoding="utf-8") as f:
    for line in sorted_singular_lines:
        f.write(line + "\n")

print(f"Готово! Результат записан в {output_file}")
