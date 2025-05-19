from tqdm import tqdm
from progress_manager import BaseProgressManager

class ConsoleProgressManager(BaseProgressManager):
    def __init__(self):
        self.main_progress = None
        self.task_progress = None

    def start_main_progress(self, total_steps):
        self.main_progress = tqdm(total=total_steps, desc="Общий прогресс", position=0)

    def start_task(self, description, total):
        if self.task_progress:
            self.task_progress.close()
        self.task_progress = tqdm(total=total, desc=description, position=1, leave=False)

    def update_main(self, value=1):
        if self.main_progress:
            self.main_progress.update(value)

    def update_task(self, value=1):
        if self.task_progress:
            self.task_progress.update(value)

    def set_task_description(self, text):
        if self.task_progress:
            self.task_progress.set_description(text)

    def close(self):
        if self.task_progress:
            self.task_progress.close()
        if self.main_progress:
            self.main_progress.close()
