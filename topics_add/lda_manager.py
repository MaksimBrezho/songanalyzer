from gensim import corpora
from gensim.models import LdaMulticore

class LDAManager:
    def __init__(self, config):
        self.config = config['lda_model']
        self.workers = config['processing']['lda_workers']

    def train(self, corpus, dictionary, progress_cb=None):
        # Создаем модель без обучения (passes=1, corpus=None)
        model = LdaMulticore(
            corpus=None,
            id2word=dictionary,
            num_topics=self.config['num_topics'],
            passes=1,  # Обучаем вручную по эпохам
            iterations=self.config['iterations'],
            alpha=self.config['alpha'],
            eta=self.config['eta'],
            chunksize=self.config['chunksize'],
            eval_every=self.config['eval_every'],
            minimum_probability=self.config['minimum_probability'],
            random_state=self.config['random_state'],
            dtype=self.config['dtype'],
            workers=self.workers,
        )

        total_passes = self.config['passes']

        for epoch in range(total_passes):
            model.update(corpus)  # Один проход обучения
            if progress_cb:
                progress_cb.update_task(1)

        return model

    @staticmethod
    def create_corpus(texts):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        return corpus, dictionary
