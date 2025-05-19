import spacy
from multiprocessing import Pool
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def _preprocess_text(self, text):
        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha
                and not token.is_stop
                and len(token.text) > self.config['text_processing']['min_token_length']]

    def process(self, meta, progress_cb=None):
        def generator():
            for entry in meta.values():
                yield entry['lyrics_en']

        results = []
        with Pool() as pool:
            for tokens in pool.imap(self._preprocess_text, generator()):
                results.append(tokens)
                if progress_cb:
                    progress_cb.update_task(1)
        return results

