import json
import os
import traceback

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QSpinBox, QDoubleSpinBox,
    QMessageBox, QSplitter, QTextBrowser, QProgressBar
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal

from topics_add.gui_progress_manager import GUIProgressManager

# Подсказки и варианты значений
TOOLTIPS = {
    "min_token_length": "Минимальная длина токена (символов).\n"
                        "Пример: 3 — игнорируются слова короче 3 символов.\n"
                        "Меньшее значение даёт больше токенов, обработка длится дольше.",
    "num_topics": "Количество тем для модели LDA.\n"
                  "Пример: 10 — выделится 10 тематик в текстах.\n"
                  "Большее число увеличивает время обучения.",
    "passes": "Количество проходов обучения LDA по всему корпусу.\n"
              "Пример: 15 — модель сделает 15 итераций прохода.\n"
              "Больше проходов — выше качество, но дольше.",
    "iterations": "Максимальное число итераций внутри одного прохода.\n"
                  "Пример: 400 — ограничение на итерации внутри прохода.",
    "alpha": "Параметр сглаживания распределения тем для документа.\n"
             "Может быть числом или строкой из ['asymmetric','symmetric','auto'].\n"
             "Пример: 'auto' — автоматический подбор.",
    "eta": "Параметр распределения слов по темам (0–1).\n"
           "Пример: 0.01 — высокая концентрация, менее размытые темы.",
    "chunksize": "Размер батча (количество документов за раз).\n"
                 "Пример: 2000 — влияет на скорость и качество обучения.",
    "eval_every": "Как часто пересчитывать перплексию.\n"
                  "Пример: 1 — после каждого прохода.\n"
                  "0 — не пересчитывать (ускорение).",
    "minimum_probability": "Минимальная вероятность темы для учёта.\n"
                           "Пример: 0.05 — игнорировать темы с вероятностью <5%.",
    "random_state": "Сид генератора случайных чисел для воспроизводимости.\n"
                    "Пример: 42 — фиксированное случайное состояние.",
    "dtype": "Тип данных для вычислений.\n"
             "Варианты: 'float32' (быстрее, меньше памяти), 'float64' (точнее).",
    "topic_words_for_naming": "Сколько слов из темы брать для генерации названия.\n"
                              "Пример: 10 — использовать 10 самых значимых слов.",
    "topic_embed_topn": "Сколько слов брать для вычисления эмбеддинга темы.\n"
                       "Пример: 15 — среднее эмбеддинг из 15 слов.",
    "sentence_model": "Модель для получения эмбеддингов предложений.\n"
                      "Варианты: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'.",
    "top_lda_topics_per_doc": "Сколько LDA-тем присваивать одному документу.\n"
                             "Пример: 3 — три наиболее вероятные темы.",
    "top_semantic_topics_per_doc": "Сколько семантических тем присваивать документу.\n"
                                  "Пример: 2 — две наиболее релевантные темы по эмбеддингу.",
    "theme_words_topk": "Количество слов для построения тематической карты.\n"
                       "Пример: 50 — топ 50 слов по частоте или значимости.",
    "topic_similarity_topk": "Сколько похожих тем учитывать при анализе.\n"
                            "Пример: 5 — топ 5 похожих тем для каждого анализа.",
    "similarity_power": "Степень усиления при подсчёте сходства слов.\n"
                       "Пример: 2 — квадратичное усиление.",
    "input_file": "Путь к входному JSON-файлу с данными.\n"
                  "Пример: 'data/songs.json'.",
    "output_file": "Имя выходного файла без расширения '.json'.\n"
                   "Пример: 'results/analysis_output'.",
    "topics_info_file": "JSON-файл для хранения информации о темах.\n"
                        "Пример: 'results/topics_info.json'.",
    "theme_words_file": "JSON-файл для карты темы → слова.\n"
                       "Пример: 'results/theme_words.json'.",
    "lda_workers": "Количество процессов (потоков) для обучения LDA.\n"
                   "Пример: 4 — использовать 4 ядра CPU.",
    "progress_manager": "Тип прогресс-менеджера.\n"
                        "Варианты: 'console' — вывод в консоль, 'gui' — графический интерфейс.",
}


CHOICES = {
    "alpha": ["asymmetric", "symmetric", "auto"],
    "dtype": ["float32", "float64"],
    "progress_manager": ["console", "gui"],
    "sentence_model": ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
}


class ConfigEditor(QWidget):
    def __init__(self, config_path):
        super().__init__()
        self.setWindowTitle("Редактор конфига")
        self.resize(1000, 800)

        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.widgets = {}

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel(f"Редактирование конфига: {config_path}"))

        splitter = QSplitter(Qt.Orientation.Vertical)

        top_layout = QVBoxLayout()
        top_widget = QWidget()
        top_widget.setLayout(top_layout)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Параметр", "Значение"])
        self.tree.setColumnWidth(0, 350)
        self.tree.itemClicked.connect(self.show_tooltip)
        top_layout.addWidget(self.tree)
        self.build_items()

        bottom_layout = QVBoxLayout()
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        self.tooltip_box = QTextBrowser()
        self.tooltip_box.setFixedHeight(80)
        self.tooltip_box.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray; padding: 4px;")
        bottom_layout.addWidget(QLabel("Подсказка:"))
        bottom_layout.addWidget(self.tooltip_box)

        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)

        main_layout.addWidget(splitter)

        btn_save = QPushButton("💾 Сохранить конфиг")
        btn_save.clicked.connect(self.save_config)
        main_layout.addWidget(btn_save)

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_items(self):
        def recurse(parent_item, subtree, path=()):
            for key, val in subtree.items():
                item = QTreeWidgetItem([key])
                parent_item.addChild(item)
                current_path = path + (key,)
                if isinstance(val, dict):
                    recurse(item, val, current_path)
                else:
                    w = self.create_widget(key, val)
                    self.tree.setItemWidget(item, 1, w)
                    self.widgets[current_path] = w
                    tip = TOOLTIPS.get(key)
                    if tip:
                        w.setToolTip(tip)
        recurse(self.tree.invisibleRootItem(), self.config)

    def create_widget(self, key, val):
        ranges = {
            "min_token_length": (1, 20),
            "num_topics": (2, 500),
            "passes": (1, 100),
            "iterations": (10, 1000),
            "eta": (0.0, 1.0),
            "chunksize": (1, 1000),
            "eval_every": (0, 10),
            "minimum_probability": (0.0, 1.0),
            "random_state": (0, 99999),
            "topic_words_for_naming": (1, 100),
            "topic_embed_topn": (1, 100),
            "top_lda_topics_per_doc": (1, 20),
            "top_semantic_topics_per_doc": (1, 20),
            "theme_words_topk": (1, 100),
            "topic_similarity_topk": (1, 20),
            "similarity_power": (1, 5),
            "lda_workers": (1, 32),
        }
        file_in = {"input_file"}
        file_out = {"output_file", "topics_info_file", "theme_words_file"}

        if key in file_in:
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit(str(val))
            btn = QPushButton("...")
            btn.clicked.connect(lambda _, le=le: self.select_file(le, True))
            lay.addWidget(le)
            lay.addWidget(btn)
            return w

        if key in file_out:
            return QLineEdit(str(val))

        if key in CHOICES:
            if key == "alpha":
                combo = QComboBox()
                combo.setEditable(True)
                combo.addItems([str(c) for c in CHOICES[key]])
                combo.setCurrentText(str(val))
                return combo
            else:
                combo = QComboBox()
                combo.addItems([str(c) for c in CHOICES[key]])
                combo.setCurrentText(str(val))
                if isinstance(val, (int, float)):
                    combo.setEditable(True)
                return combo

        if key in ranges:
            lo, hi = ranges[key]
            if isinstance(val, float):
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setDecimals(3)
                sb.setSingleStep((hi - lo) / 100)
                sb.setValue(val)
                return sb
            if isinstance(val, int):
                sb = QSpinBox()
                sb.setRange(lo, hi)
                sb.setValue(val)
                return sb
            if isinstance(val, str):
                try:
                    fv = float(val)
                    sb = QDoubleSpinBox()
                    sb.setRange(lo, hi)
                    sb.setDecimals(3)
                    sb.setValue(fv)
                    return sb
                except:
                    return QLineEdit(val)

        return QLineEdit(str(val))

    def select_file(self, line_edit: QLineEdit, existing: bool):
        cwd = os.getcwd()
        if existing:
            fname, _ = QFileDialog.getOpenFileName(self, "Выбрать файл", cwd, "JSON (*.json)")
        else:
            fname, _ = QFileDialog.getSaveFileName(self, "Сохранить как", cwd, "JSON (*.json)")
        if fname:
            rel = os.path.relpath(fname, cwd)
            line_edit.setText(rel)

    def save_config(self):
        try:
            new_conf = self.collect_config(self.config, ())
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(new_conf, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "Успех", "Конфиг сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def collect_config(self, ref: dict, path: tuple):
        result = {}
        for key, val in ref.items():
            curr = path + (key,)
            if isinstance(val, dict):
                result[key] = self.collect_config(val, curr)
            else:
                w = self.widgets.get(curr)
                result[key] = self.get_value(w) if w else val
        return result

    def get_value(self, widget):
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            txt = widget.currentText()
            try:
                return float(txt)
            except ValueError:
                return txt
        if isinstance(widget, QWidget):
            inner = widget.findChild(QLineEdit)
            if inner:
                return inner.text()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def show_tooltip(self, item):
        key = item.text(0)
        tip = TOOLTIPS.get(key, "")
        self.tooltip_box.setPlainText(tip or "Нет описания")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тематический анализ текстов")
        self.resize(1200, 800)

        # Прогресс-бары и статус
        self.main_progress_bar = QProgressBar()
        self.task_progress_bar = QProgressBar()
        self.status_label = QLabel("Готов к работе")

        # Редактор конфига
        self.config_editor = ConfigEditor("config.json")

        # Кнопка запуска
        self.btn_run = QPushButton("▶ Запустить обработку")
        self.btn_run.clicked.connect(self.run_processing)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.config_editor)

        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Общий прогресс:"))
        progress_layout.addWidget(self.main_progress_bar)
        progress_layout.addWidget(QLabel("Текущая задача:"))
        progress_layout.addWidget(self.task_progress_bar)
        progress_layout.addWidget(self.status_label)

        layout.addLayout(progress_layout)
        layout.addWidget(self.btn_run)
        self.setLayout(layout)

        # Инициализация менеджера прогресса
        self.progress_manager = GUIProgressManager(
            self.main_progress_bar,
            self.task_progress_bar,
            self.status_label
        )

    def show_error(self, msg):
        QMessageBox.critical(self, "Ошибка", msg)

    def run_processing(self):
        # Запуск в отдельном потоке
        self.thread = QThread()
        self.worker = ProcessingWorker()
        self.worker.error.connect(self.show_error)  # Подключение сигнала ОШИБКИ
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

class ProcessingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            from main import main
            main()
        except Exception as e:
            error_msg = f"Ошибка:\n{str(e)}\n\nТрассировка:\n{traceback.format_exc()}"
            self.error.emit(error_msg)  # Отправляем сообщение об ошибке
        finally:
            self.finished.emit()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # приятный стиль по умолчанию
    editor = ConfigEditor("config.json")
    editor.show()
    sys.exit(app.exec())
