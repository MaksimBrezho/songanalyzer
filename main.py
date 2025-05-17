from difflib import SequenceMatcher

from song_loader import SongDataLoader
from genius_downloader import GeniusLyricsDownloader
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download song lyrics from Genius')
    parser.add_argument('--input', type=str, default='songs.csv', help='Input file with songs')
    parser.add_argument('--output', type=str, default='lyrics2.json', help='Output JSON file')
    parser.add_argument('--token', type=str, default='6TqyEn9FV3x5E-bo77RO3a70CPlH971KHJQlhgIvrUM7flApF5qmFLFwOBSuKJhr', help='Genius API token')
    parser.add_argument('--limit', type=int, help='Maximum number of songs to process')
    parser.add_argument('--retries', type=int, default=5, help='Max retry attempts')
    args = parser.parse_args()

    # Load songs
    songs = SongDataLoader().load_songs(args.input, limit=args.limit)
    print(f"Loaded {len(songs)} songs from {args.input}")

    # Initialize downloader
    downloader = GeniusLyricsDownloader(
        api_token=args.token,
        max_retries=args.retries,
        request_timeout=30.0
    )

    # Download lyrics
    results = downloader.download_lyrics(
        songs,
        output_file=args.output,
        skip_existing=True
    )

    # Print summary
    success = sum(1 for v in results.values() if v['found'])
    print(f"\nDownload complete!\nSuccess: {success}\nFailed: {len(results) - success}")

if __name__ == "__main__":
    print(SequenceMatcher(None, "The Prodigy".lower(), "T he Prodigy".lower()).ratio())
    main()