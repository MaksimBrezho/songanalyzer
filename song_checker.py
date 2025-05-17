import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import copy
import os


class SongProcessor:
    def __init__(self, master, processed_data):
        self.master = master
        self.processed_data = processed_data
        self.set_auto_approved()

        self.master.title("Song Processor")
        self.master.geometry("1000x800")

        self.container = ttk.Frame(self.master)
        self.container.pack(fill='both', expand=True)

        self.load_data()

    def set_auto_approved(self):
        """Автоматически помечаем только ABSOLUTELY EXACT записи как approved"""
        for key in self.processed_data:
            entry = self.processed_data[key]
            if (entry.get('match_quality') == 'ABSOLUTELY EXACT' or
                    entry.get('match_quality') == 'EXACT' or
                    entry.get('match_quality') == 'VERY CLOSE'):
                entry['approved'] = True

    def parse_key(self, key):
        """Разбивает ключ на автора и название"""
        if '_' in key:
            parts = key.split('_', 1)
            return (
                self.format_title_case(parts[0]),  # artist
                self.format_title_case(parts[1])  # title
            )
        return ('', '')

    def load_data(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        self.notebook = ttk.Notebook(self.container)
        self.notebook.pack(fill='both', expand=True)

        for key in list(self.processed_data.keys()):
            entry = self.processed_data[key]

            # Пропускаем записи с approved=True или ABSOLUTELY EXACT
            if entry.get('approved') or (entry.get('match_quality') == 'ABSOLUTELY EXACT' or
                    entry.get('match_quality') == 'EXACT' or
                    entry.get('match_quality') == 'VERY CLOSE'):
                continue

            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=key)

            if not entry['found']:
                self.create_not_found_ui(frame, key)
            else:
                if entry.get('no_lyrics'):
                    self.create_instrumental_ui(frame, key)
                else:
                    self.create_review_ui(frame, key)

    def create_not_found_ui(self, parent, key):
        # Парсим ключ для автора и названия
        default_artist, default_title = self.parse_key(key)

        # Создаем элементы интерфейса
        ttk.Label(parent, text=f"Key: {key}").pack(pady=5)

        # Поле для названия
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill='x', pady=5)
        ttk.Label(title_frame, text="Title:").pack(side='left')
        title_entry = ttk.Entry(title_frame)
        title_entry.insert(0, default_title)
        title_entry.pack(side='left', expand=True, fill='x')

        # Поле для автора
        artist_frame = ttk.Frame(parent)
        artist_frame.pack(fill='x', pady=5)
        ttk.Label(artist_frame, text="Artist:").pack(side='left')
        artist_entry = ttk.Entry(artist_frame)
        artist_entry.insert(0, default_artist)
        artist_entry.pack(side='left', expand=True, fill='x')

        # Кнопки и остальные элементы
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Mark as Instrumental",
                 command=lambda: self.add_instrumental(
                     key,
                     title_entry.get().strip(),
                     artist_entry.get().strip()
                 )).pack(side='left', padx=5)

        lyrics_text = scrolledtext.ScrolledText(parent, height=10)
        lyrics_text.pack(fill='both', expand=True)

        ttk.Button(btn_frame, text="Add Lyrics",
                 command=lambda: self.add_lyrics(
                     key,
                     artist_entry.get().strip(),
                     title_entry.get().strip(),
                     lyrics_text.get("1.0", tk.END).strip()
                 )).pack(side='left', padx=5)

        ttk.Button(btn_frame, text="Delete",
                 command=lambda: self.delete_entry(key)).pack(side='left', padx=5)

    def add_instrumental(self, key, title, artist):
        if not title or not artist:
            messagebox.showerror("Error", "Please fill title and artist fields")
            return

        self.processed_data[key].update({
            'found': True,
            'title': title,
            'artist': artist,
            'no_lyrics': True,
            'approved': True
        })
        self.save_data()
        self.load_data()

    def create_review_ui(self, parent, key):
        entry = self.processed_data[key]

        match_quality = entry.get('match_quality', 'N/A')
        similarity_score = entry.get('similarity_score', 'N/A')
        original_title = entry.get('original_title', 'N/A')
        original_artist = entry.get('original_artist', 'N/A')
        title = entry.get('title', 'N/A')
        artist = entry.get('artist', 'N/A')

        ttk.Label(parent,
                  text=f"Match Quality: {match_quality} ({similarity_score})"
                  ).pack(pady=5)

        ttk.Label(parent, text=f"Original Title: {original_title}").pack()
        ttk.Label(parent, text=f"Original Artist: {original_artist}").pack()
        ttk.Label(parent, text=f"Proposed Title: {title}").pack()
        ttk.Label(parent, text=f"Proposed Artist: {artist}").pack()

        lyrics_text = scrolledtext.ScrolledText(parent, height=10)
        lyrics_text.insert(tk.END, entry['lyrics'])
        lyrics_text.config(state=tk.DISABLED)
        lyrics_text.pack(fill='both', expand=True)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Approve",
                   command=lambda: self.approve_entry(key)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Edit Lyrics",
                   command=lambda: self.edit_lyrics(key)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Mark as Instrumental",
                   command=lambda: self.mark_as_instrumental(key)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Delete",
                   command=lambda: self.delete_entry(key)).pack(side='left', padx=5)

    def mark_as_instrumental(self, key):
        """Обработка для найденных записей"""
        entry = self.processed_data[key]
        entry.update({
            'title': entry['original_title'],
            'artist': entry['original_artist'],
            'no_lyrics': True,
            'approved': True
        })
        # Удаляем лишние поля
        for field in ['lyrics', 'genius_id', 'url', 'similarity_score', 'match_quality']:
            entry.pop(field, None)
        self.save_data()
        self.load_data()

    def delete_entry(self, key):
        if messagebox.askyesno("Confirm", f"Delete entry '{key}'?"):
            del self.processed_data[key]
            self.save_data()
            self.load_data()

    def format_title_case(self, s):
        """Форматирует строку в Title Case с учетом апострофов"""
        formatted_words = []
        for word in s.split():
            if word:
                formatted = word[0].upper() + word[1:].lower()
                formatted_words.append(formatted)
        return ' '.join(formatted_words)

    def create_instrumental_ui(self, parent, key):
        entry = self.processed_data[key]
        ttk.Label(parent, text="INSTRUMENTAL VERSION", style='Info.TLabel').pack(pady=20)
        ttk.Label(parent, text=f"Title: {entry['title']}").pack()
        ttk.Label(parent, text=f"Artist: {entry['artist']}").pack()

    def add_lyrics(self, key, title, artist, lyrics):
        if not title or not artist:
            messagebox.showerror("Error", "Please fill title and artist fields")
            return

        self.processed_data[key].update({
            'title': title,
            'artist': artist,
            'lyrics': lyrics.strip(),
            'found': True,
            'approved': True
        })
        self.save_data()
        self.load_data()

    def approve_entry(self, key):
        self.processed_data[key]['approved'] = True
        self.save_data()
        self.load_data()

    def save_edited_lyrics(self, key, lyrics, window):
        self.processed_data[key]['lyrics'] = lyrics.strip()
        self.processed_data[key]['approved'] = True
        window.destroy()
        self.save_data()
        self.load_data()

    def edit_lyrics(self, key):
        edit_win = tk.Toplevel()
        edit_win.title("Edit Lyrics")

        text_area = scrolledtext.ScrolledText(edit_win, height=20)
        text_area.insert(tk.END, self.processed_data[key]['lyrics'])
        text_area.pack(padx=10, pady=10)

        ttk.Button(edit_win, text="Save",
                   command=lambda: self.save_edited_lyrics(key, text_area.get("1.0", tk.END), edit_win)
                   ).pack(pady=5)

    def save_data(self):
        """Автосохранение после каждого изменения"""
        # Сохраняем все записи, включая автоматически подтвержденные
        with open('processed_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Пытаемся загрузить предыдущий прогресс
    if os.path.exists('processed_data.json'):
        with open('processed_data.json', 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
    else:
        with open('lyrics2.json', 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            processed_data = copy.deepcopy(original_data)

    root = tk.Tk()
    style = ttk.Style()
    style.configure('Success.TLabel', foreground='green')
    style.configure('Info.TLabel', foreground='blue')
    app = SongProcessor(root, processed_data)
    root.mainloop()