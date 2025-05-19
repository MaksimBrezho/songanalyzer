# progress_manager.py
from abc import ABC, abstractmethod


class BaseProgressManager(ABC):
    @abstractmethod
    def start_main_progress(self, total_steps): pass

    @abstractmethod
    def start_task(self, description, total): pass

    @abstractmethod
    def update_main(self, value=1): pass

    @abstractmethod
    def update_task(self, value=1): pass

    @abstractmethod
    def set_task_description(self, text): pass

    @abstractmethod
    def close(self): pass