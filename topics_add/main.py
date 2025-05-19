# main.py
import multiprocessing
import time

from PyQt6.QtWidgets import QApplication, QProgressBar, QLabel

from data_loader import DataLoader
from text_preprocessor import TextPreprocessor
from lda_manager import LDAManager
from topic_namer import TopicNamer
from embedding_generator import EmbeddingGenerator
from progress_manager import BaseProgressManager
from console_progress_manager import ConsoleProgressManager
from gui_progress_manager import GUIProgressManager
from sentence_transformers import SentenceTransformer, util
from lda_topic_namer_with_embeddings import load_candidate_names


class TopicAssigner:
    @staticmethod
    def assign_lda_topics(meta, corpus, lda_model, topic_names, topic_scores, config, progress_manager):
        def process_args():
            for (key, entry), doc_bow in zip(meta.items(), corpus):
                yield (key, entry, doc_bow, topic_names, topic_scores, lda_model, config)

        progress_manager.start_task("Присвоение LDA тем", len(meta))
        results = []

        with multiprocessing.Pool() as pool:
            args = list(process_args())
            for result in pool.imap(TopicAssigner._assign_lda_topic, args):
                results.append(result)
                progress_manager.update_task()

        progress_manager.update_main()
        return dict(results)

    @staticmethod
    def _assign_lda_topic(args):
        key, entry, doc_bow, topic_names, topic_scores, lda, config = args
        topic_distribution = lda.get_document_topics(doc_bow)
        top_topics = sorted(topic_distribution, key=lambda x: -x[1])[
                     :config['topic_assignment']['top_lda_topics_per_doc']]

        entry['lda_topics'] = [{
            'topic': t[0],
            'prob': float(t[1]),
            'name': topic_names[t[0]],
            'score': topic_scores[t[0]]
        } for t in top_topics]

        return key, entry

    @staticmethod
    def assign_semantic_topics(meta, topic_embeddings, topic_names, config, embedding_generator, progress_manager):
        def process_args():
            for key, entry in meta.items():
                yield (key, entry, topic_embeddings, topic_names, config, embedding_generator)

        progress_manager.start_task("Присвоение семантических тем", len(meta))
        results = []

        with multiprocessing.Pool() as pool:
            args = list(process_args())
            for result in pool.imap(TopicAssigner._assign_semantic_topic, args):
                results.append(result)
                progress_manager.update_task()

        progress_manager.update_main()
        return dict(results)

    @staticmethod
    def _assign_semantic_topic(args):
        key, entry, topic_embeddings, topic_names, config, embedding_generator = args
        lyrics = entry.get("lyrics_en", "")
        doc_embedding = embedding_generator.get_document_embedding(lyrics)

        similarities = [
            (tid, util.cos_sim(doc_embedding, emb).item())
            for tid, emb in topic_embeddings.items()
        ]
        top_topics = sorted(similarities, key=lambda x: -x[1])[
                     :config['topic_assignment']['top_semantic_topics_per_doc']]

        entry['semantic_topics'] = [{
            'topic': tid,
            'similarity': float(sim),
            'name': topic_names[tid]
        } for tid, sim in top_topics]

        return key, entry


class SimilarityCalculator:
    @staticmethod
    def calculate_theme_similarities(meta, theme_to_words, config, preprocessor, progress_manager):
        def calc_score(text_tokens, topic_words_sim):
            topic_word_dict = {w: sim for w, sim in topic_words_sim}
            return sum(pow(topic_word_dict.get(token, 0), config['similarity']['similarity_power'])
                       for token in text_tokens)

        progress_manager.start_task("Расчет схожести с темами", len(meta))

        for key, entry in meta.items():
            text = entry.get('lyrics_en', '').lower()
            doc_tokens = preprocessor._preprocess_text(text)
            scores = [(theme, calc_score(doc_tokens, words))
                      for theme, words in theme_to_words.items()]
            entry['topic_word_similarity_scores'] = [{
                "topic": t[0],
                "score": t[1]
            } for t in sorted(scores, key=lambda x: -x[1])[:5]]

            progress_manager.update_task()

        progress_manager.update_main()
        return meta


def main():
    data_loader = DataLoader()
    config = data_loader.load_config('config.json')

    # Автоматический выбор GUI/Console
    if QApplication.instance() is not None:
        from gui import MainWindow
        app = QApplication.instance()
        window = MainWindow()
        progress_manager = window.progress_manager
    else:
        pm_type = config.get('processing', {}).get('progress_manager', 'console').lower()
        if pm_type == 'gui':
            # Для GUI используйте реальные прогресс-бары, а не фиктивные
            # (этот код должен быть в классе MainWindow)
            raise NotImplementedError("GUI режим требует запуска через run_gui.py")
        else:
            progress_manager = ConsoleProgressManager()  # Только консольный режим

    progress_manager.start_main_progress(11)

    try:
        # 1. Загрузка метаданных
        progress_manager.start_task("Загрузка метаданных", 1)
        meta = data_loader.load_meta(config['files']['input_file'])
        time.sleep(1)  # задержка, чтобы увидеть прогрессбар
        progress_manager.update_task()
        progress_manager.update_main()

        # 2. Предобработка текстов
        progress_manager.start_task("Предобработка текстов", len(meta))
        preprocessor = TextPreprocessor(config)
        texts = preprocessor.process(meta, progress_manager)
        progress_manager.update_main()

        # 3. Обучение LDA
        progress_manager.start_task("Обучение LDA модели", config['lda_model']['passes'])
        lda_manager = LDAManager(config)
        corpus, dictionary = lda_manager.create_corpus(texts)
        lda_model = lda_manager.train(corpus, dictionary, progress_manager)
        progress_manager.update_main()

        # 4. Наименование тем
        progress_manager.start_task("Создание названий тем", lda_model.num_topics)
        model = SentenceTransformer(config['embedding']['sentence_model'])
        topic_namer = TopicNamer(config, model)
        topic_names, topic_scores = topic_namer.name_topics(lda_model, progress_manager)
        progress_manager.update_main()

        # 5. Создание информации о темах и сохранение
        progress_manager.start_task("Создание информации о темах", 1)
        topics_info = [{
            "topic_id": tid,
            "name": topic_names[tid],
            "score": topic_scores[tid],
            "words": [word for word, _ in lda_model.show_topic(tid, config['topic_naming']['topic_words_for_naming'])]
        } for tid in range(lda_model.num_topics)]
        data_loader.save_json(topics_info, config['files']['topics_info_file'])
        time.sleep(1)  # задержка, чтобы увидеть прогрессбар
        progress_manager.update_task()
        progress_manager.update_main()

        # 6. Подбор слов к темам
        progress_manager.start_task("Подбор слов к темам", 1)
        candidate_names = load_candidate_names('candidate_names.txt')
        theme_to_words = topic_namer.find_top_words_for_themes(candidate_names, dictionary)
        data_loader.save_json(theme_to_words, config['files']['theme_words_file'])
        progress_manager.update_main()

        # 7. Присвоение LDA-тем
        meta = TopicAssigner.assign_lda_topics(
            meta, corpus, lda_model,
            topic_names, topic_scores, config, progress_manager
        )

        # 8. Генерация эмбеддингов тем
        progress_manager.start_task("Генерация векторов тем", 1)
        embedding_generator = EmbeddingGenerator(config)
        topic_embeddings = embedding_generator.get_topic_embeddings(topic_names, lda_model)
        progress_manager.update_main()

        # 9. Присвоение семантических тем
        meta = TopicAssigner.assign_semantic_topics(
            meta, topic_embeddings,
            topic_names, config,
            embedding_generator, progress_manager
        )

        # 10. Расчет схожести с темами
        meta = SimilarityCalculator.calculate_theme_similarities(
            meta, theme_to_words,
            config, preprocessor, progress_manager
        )

        # 11. Сохранение результатов
        progress_manager.start_task("Сохранение результатов", 1)
        data_loader.save_json(meta, config['files']['output_file'])
        progress_manager.update_main()

        print("\nГотово! Результаты сохранены в:")
        print(f"- Метаданные с темами: {config['files']['output_file']}")
        print(f"- Информация о темах: {config['files']['topics_info_file']}")
        print(f"- Сопоставление тем и слов: {config['files']['theme_words_file']}")

    finally:
        progress_manager.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
