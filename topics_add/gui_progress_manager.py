# gui_progress_manager.py
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QProgressBar, QLabel
from progress_manager import BaseProgressManager
from abc import ABCMeta

class MetaQObjectABC(type(QObject), ABCMeta):
    pass

class GUIProgressManager(QObject, BaseProgressManager, metaclass=MetaQObjectABC):
    update_signal = pyqtSignal(str, int, int)  # description, current, total

    def __init__(self, main_progress_bar, task_progress_bar, status_label):
        super().__init__()
        self.main_progress = main_progress_bar
        self.task_progress = task_progress_bar
        self.status_label = status_label
        self.update_signal.connect(self._handle_update)

    def start_main_progress(self, total_steps):
        self.main_progress.setMaximum(total_steps)
        self.main_progress.setValue(0)

    def start_task(self, description, total):
        self.update_signal.emit(description, 0, total)

    def update_main(self, value=1):
        new_value = self.main_progress.value() + value
        self.update_signal.emit("Общий прогресс", new_value, self.main_progress.maximum())

    def update_task(self, value=1):
        new_value = self.task_progress.value() + value
        self.update_signal.emit(self.status_label.text(), new_value, self.task_progress.maximum())

    def set_task_description(self, text):
        self.update_signal.emit(text, self.task_progress.value(), self.task_progress.maximum())

    def _handle_update(self, description, current, total):
        if description == "Общий прогресс":
            self.main_progress.setValue(current)
        else:
            self.task_progress.setMaximum(total)
            self.task_progress.setValue(current)
            self.status_label.setText(description)

    def close(self):
        self.main_progress.reset()
        self.task_progress.reset()
        self.status_label.clear()