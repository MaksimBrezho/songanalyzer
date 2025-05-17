import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class SongDataLoader:
    def __init__(self):
        self.parsers = {
            '.csv': self._parse_csv,
            '.txt': self._parse_txt
        }

    def load_songs(
            self,
            file_path: str,
            limit: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        path = Path(file_path)
        self._validate_file(path)
        parser = self._get_parser(path.suffix)
        songs = parser(path)
        return self._apply_limit(songs, limit)

    def _validate_file(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")

    def _get_parser(self, suffix: str):
        parser = self.parsers.get(suffix.lower())
        if not parser:
            raise ValueError(f"Unsupported file format: {suffix}")
        return parser

    def _parse_csv(self, path: Path) -> List[Tuple[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            return self._extract_fields(csv.DictReader(f, dialect=dialect))

    def _parse_txt(self, path: Path) -> List[Tuple[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            delimiter = self._detect_delimiter(f.readline())
            f.seek(0)
            return self._extract_fields(csv.DictReader(f, delimiter=delimiter))

    def _detect_delimiter(self, line: str) -> str:
        return '|' if '|' in line else '\t'

    def _extract_fields(self, reader) -> List[Tuple[str, str]]:
        field_map = {
            'track': ['Track name', 'title', 'song'],
            'artist': ['Artist name', 'artist', 'author']
        }
        return [
            (self._get_field(row, field_map['track']),
             self._get_field(row, field_map['artist']))
            for row in reader
            if self._get_field(row, field_map['track'])
               and self._get_field(row, field_map['artist'])
        ]

    def _get_field(self, row: Dict[str, Any], keys: List[str]) -> str:
        for key in keys:
            if key in row:
                return row[key].strip()
        return ''

    def _apply_limit(self, songs: List, limit: Optional[int]) -> List:
        return songs[:limit] if limit and limit > 0 else songs