import json
import time
import logging
import lyricsgenius
import re
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from proxy_manager import ProxyManager
from difflib import SequenceMatcher


class GeniusLyricsDownloader:
    def __init__(
            self,
            api_token: str,
            max_retries: int = 5,
            request_timeout: float = 20.0,
            base_delay: float = 1.0,
            backoff_factor: float = 2.0
    ):
        self.genius = self._init_genius(api_token, request_timeout)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.logger = self._setup_logger()
        self.proxy_manager = ProxyManager()
        self.current_proxy = None

    def _init_genius(self, token: str, timeout: float) -> lyricsgenius.Genius:
        genius = lyricsgenius.Genius(
            token,
            verbose=False,
            skip_non_songs=True,
            remove_section_headers=True,
            timeout=timeout
        )
        genius.verbose = False
        return genius

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('GeniusDownloader')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def download_lyrics(
            self,
            songs: List[Tuple[str, str]],
            output_file: str = 'lyrics.json',
            skip_existing: bool = True
    ) -> Dict[str, Any]:
        results = self._load_existing_data(output_file) if skip_existing else {}

        for idx, (title, artist) in enumerate(songs, 1):
            song_id = f"{artist.lower()}_{title.lower()}"
            if song_id in results:
                self.logger.info(f"Skipping existing: {artist} - {title}")
                continue

            result = self._fetch_song(artist, title)
            results[song_id] = result
            self._save_results(results, output_file)

            self._log_progress(idx, len(songs))
            time.sleep(self.base_delay)
        return results

    #def _verify_result(self, result: dict, original_artist: str, original_title: str) -> dict:
        #if result['found']:
        #    artist_similarity = self._similarity(original_artist, result['artist'])
        #    title_similarity = self._similarity(original_title, result['title'])

        #    # Отбрасываем результат, если схожесть слишком низкая
        #    if artist_similarity < 0.5 or title_similarity < 0.5:
        #        self.logger.warning(
        #            f"Rejected: Got '{result['artist']} - {result['title']}' "
        #            f"instead of '{original_artist} - {original_title}' "
        #            f"(artist_sim={artist_similarity:.2f}, title_sim={title_similarity:.2f})"
        #        )
        #        return {
        #            'found': False,
        #            'error': 'Low similarity',
        #            'original_artist': original_artist,
        #            'original_title': original_title
        #        }
        #
        #return result

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Схожесть между строками на основе SequenceMatcher"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _fetch_song(self, artist: str, title: str) -> Dict[str, Any]:
        #clean_artist = self._sanitize_query(artist)
        #clean_title = self._sanitize_query(title)

        best_result = None
        best_score = 0.0

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.current_proxy:
                    proxies = {
                        "http": f"http://{self.current_proxy}",
                        "https": f"https://{self.current_proxy}",
                    }
                    self.genius._session.proxies = proxies

                # Основной поиск по оригинальным данным
                song = self.genius.search_song(title, artist)
                if song:
                    result = {
                        'found': True,
                        'artist': song.artist,
                        'title': song.title,
                        'lyrics': self._clean_lyrics(song.lyrics),
                        'genius_id': song.id,
                        'url': song.url,
                        'original_artist': artist,
                        'original_title': title
                    }
                    #verified = self._verify_result(result, artist, title)
                    #if verified['found']:
                    avg_similarity = (
                                                 self._similarity(artist, result['artist']) +
                                                 self._similarity(title, result['title'])
                                         ) / 2
                    best_result = result
                    best_score = avg_similarity

                # Альтернативный поиск по вариантам
                if best_score < 1:
                    fallback = self._try_get_lyrics(artist, title)
                    if fallback:
                        fallback_score = (
                                             self._similarity(artist, fallback['artist']) +
                                             self._similarity(title, fallback['title'])
                                     ) / 2
                        if fallback_score > best_score:
                            best_result = fallback
                            best_score = fallback_score

                if best_result:
                    best_result['similarity_score'] = round(best_score, 3)
                    best_result['match_quality'] = self._match_quality_label(best_score)
                    return best_result

                return {'found': False, 'error': 'No matching result found'}

            except Exception as e:
                error = str(e)
                delay = self.base_delay * (self.backoff_factor ** attempt)
                self.logger.warning(f"Attempt {attempt} failed. Error: {error}")
                self.current_proxy = self.proxy_manager.get_proxy()
                if self.current_proxy:
                    self.logger.info(f"Switching to new proxy: {self.current_proxy}")
                time.sleep(delay)

        return {'found': False, 'error': 'Max retries exceeded'}

    def _try_get_lyrics(self, original_artist: str, original_title: str) -> Optional[Dict[str, Any]]:
        title_variants = self._generate_title_variants(original_title)
        artist_variants = self._generate_artist_variants(original_artist)

        best_result = None
        best_score = 0.0

        for artist in artist_variants:
            for title in title_variants:
                try:
                    song = self.genius.search_song(title, artist)
                    if song:
                        artist_sim = self._similarity(original_artist, song.artist)
                        title_sim = self._similarity(original_title, song.title)
                        avg_sim = (artist_sim + title_sim) / 2
                        if avg_sim > best_score:
                            best_score = avg_sim
                            best_result = {
                                'found': True,
                                'artist': song.artist,
                                'title': song.title,
                                'lyrics': self._clean_lyrics(song.lyrics),
                                'genius_id': song.id,
                                'url': song.url,
                                'original_artist': original_artist,
                                'original_title': original_title
                            }
                except Exception as e:
                    self.logger.warning(f"Error during fallback search for {artist} - {title}: {e}")
                    continue

        return best_result if best_result else None

    def _match_quality_label(self, score: float) -> str:
        if score < 0.5:
            return "UNRELIABLE"
        elif score < 0.7:
            return "SIMILAR"
        elif score < 0.9:
            return "VERY CLOSE"
        elif score < 0.96:
            return "EXACT"
        else:
            return "ABSOLUTELY EXACT"

    def _search_lyrics(self, artist: str, title: str) -> Optional[str]:
        """
        Выполняет реальный поиск текста — например, по API или парсингом.
        """
        try:
            song = self.genius.search_song(title, artist)
            if song:
                return song.lyrics
        except Exception as e:
            self.logger.warning(f"Error during lyrics search: {e}")
        return None

    def _generate_alternatives(self, artist: str, title: str) -> List[Tuple[str, str]]:
        """
        Генерирует постепенно упрощённые варианты названия и исполнителя.
        """
        alt_titles = self._generate_title_variants(title)
        alt_artists = self._generate_artist_variants(artist)
        return [(a, t) for a in alt_artists for t in alt_titles]

    def _generate_title_variants(self, title: str) -> List[str]:
        variants = set()
        title = title.strip()
        variants.add(title)

        # Удаление (Remastered ...), (Soundtrack Version), etc.
        no_parens = re.sub(r'\s*[\[(][^)\]]*(remaster(ed)?|soundtrack|version)[^)\]]*[\])]', '', title,
                           flags=re.IGNORECASE)
        variants.add(no_parens.strip())

        # Удаление "- Remastered 2021" или "- Soundtrack Version"
        no_suffix = re.sub(r'\s*-\s*(remaster(ed)?( \d{4})?|soundtrack version)', '', title, flags=re.IGNORECASE)
        variants.add(no_suffix.strip())

        # Удаление "from ... soundtrack"
        no_from_soundtrack = re.sub(r'\s*from\s+.*\bsoundtrack\b.*$', '', title, flags=re.IGNORECASE)
        variants.add(no_from_soundtrack.strip())

        # Удаление "feat", "ft", "vs", "with", "remix", "prod by" и прочего
        no_feature = re.sub(r'\s*(ft|feat|vs|with|aka|prod|remix)[^,;\-]*$', '', title, flags=re.IGNORECASE)
        variants.add(no_feature.strip())

        # Удаление всех неалфавитных символов (кроме пробелов и тире)
        no_symbols = re.sub(r'[^\w\s-]', '', title)
        variants.add(no_symbols.strip())

        # Возвращаем уникальные и отсечённые до 50 символов
        return list(dict.fromkeys([v[:50] for v in variants if v]))

    def _generate_artist_variants(self, artist: str) -> List[str]:
        variants = set()
        artist = artist.strip()
        variants.add(artist)

        # Удаление "feat", "ft", "vs", "aka", "with", "and" и прочего
        no_feature = re.sub(r'\s*(ft|feat|vs|aka|with|and)\s+.*$', '', artist, flags=re.IGNORECASE)
        variants.add(no_feature.strip())

        # Удаление содержимого скобок
        no_parens = re.sub(r'[\[(].*?[\])]', '', artist)
        variants.add(no_parens.strip())

        # Удаление символов, кроме слов, пробелов и тире
        no_symbols = re.sub(r'[^\w\s-]', '', artist)
        variants.add(no_symbols.strip())

        return list(dict.fromkeys([v[:50] for v in variants if v]))

    #def _sanitize_query(self, text: str) -> str:
    #    text = re.sub(r'\s*[([{].*?[)}\]]', '', text)  # Удаление скобок и их содержимого
    #    text = re.sub(r'\s*(?:ft|feat|vs|with|aka|prod|remix).*', '', text, flags=re.IGNORECASE)
    #    text = re.sub(r'[^\w\s-]', '', text).strip()
    #    return text[:50]  # Ограничение длины запроса

    def _load_existing_data(self, output_file: str) -> Dict[str, Any]:
        if Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_results(self, data: Dict[str, Any], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_progress(self, current: int, total: int):
        progress = current / total * 100
        self.logger.info(f"Processed: {current}/{total} ({progress:.1f}%)")

    def _clean_lyrics(self, lyrics: str) -> str:
        # Удаляем всё до и включая "Lyrics"
        lyrics = re.sub(r'^.*?Lyrics', '', lyrics, flags=re.DOTALL | re.IGNORECASE)

        # Удаляем всё до и включая "… Read More"
        lyrics = re.sub(r'^.*?… Read More', '', lyrics, flags=re.DOTALL)

        return lyrics.strip()
