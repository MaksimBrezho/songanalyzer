import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import time
from tqdm import tqdm
from datetime import datetime


def get_subcategories_themes(base_url, start_category, max_depth=4, delay=0):
    visited = set()
    themes = set()
    base_domain = urljoin(base_url, '/wiki/')
    progress_bar = tqdm(desc="Processing categories", unit="cat")

    pattern = re.compile(
        r'(?:'
        r'^Category:|'
        r'Songs about |'
        r'Songs by |'
        r'Songs |'
        r'Lists of a |'
        r'Lists of |'
        r'Ballads by |'
        r'Ballads of |'
        r'Songs with |'
        r'songs$|'
        r'song$|'
        r' by theme$|'
        r'Works about |'
        r'\s*\(.*\)|'
        r'\s+themes?|'
        r'\s+music|'
        r'\s+ballads|'
        r'^Works |'
        r'^Pages |'
        r'^Articles'
        r')',
        flags=re.IGNORECASE
    )

    def process_category(url, depth):
        nonlocal progress_bar
        if depth > max_depth or url in visited:
            return

        visited.add(url)
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"Depth: {depth}, URL: {url.split('/')[-1][:15]}...")
        time.sleep(delay)

        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
        except Exception as e:
            progress_bar.write(f"Error fetching {url}: {e}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')

        category_links = soup.select('#mw-subcategories a[href^="/wiki/Category:"]')
        for link in category_links:
            title = link.get_text(strip=True)
            cleaned = pattern.sub('', title).strip()
            if cleaned and len(cleaned) > 2:
                themes.add(cleaned)

        if depth < max_depth:
            subcategories = soup.select('#mw-subcategories a[href^="/wiki/Category:"]')
            for subcat in subcategories:
                subcat_url = urljoin(base_url, subcat['href'])
                if subcat_url not in visited:
                    process_category(subcat_url, depth + 1)

    try:
        start_url = urljoin(base_url, f'/wiki/Category:{start_category}')
        process_category(start_url, 0)
    finally:
        progress_bar.close()

    return sorted(themes)


def save_to_file(themes, filename=None):
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"song_themes_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for i, theme in enumerate(themes, 1):
                f.write(f"{theme}\n")
        return filename
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        return None


if __name__ == "__main__":
    wikipedia_url = "https://en.wikipedia.org"
    main_category = "Songs_by_theme"

    print(f"Собираем темы из категории {main_category}...\n")
    themes = get_subcategories_themes(wikipedia_url, main_category)

    print("\nРезультаты:")
    print(f"Найдено уникальных тем: {len(themes)}")

    # Сохраняем в файл
    saved_file = save_to_file(themes)
    if saved_file:
        print(f"\nРезультаты сохранены в файл: {saved_file}")
    else:
        print("\nНе удалось сохранить результаты в файл")