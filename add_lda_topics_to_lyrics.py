import json
import spacy
import torch
import argparse
import multiprocessing
from gensim import corpora
from gensim.models import LdaMulticore
from tqdm import tqdm
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from lda_topic_namer_with_embeddings import get_topic_name, load_candidate_names
from gensim.models.callbacks import CoherenceMetric, ConvergenceMetric
from gensim import corpora, models
from tqdm.auto import tqdm

class ProgressTracker:
    def __init__(self, total_passes):
        self.pbar = tqdm(total=total_passes, desc="LDA Training")
        self.epoch = 0

    def __call__(self, _, logger, corpus):
        self.epoch += 1
        self.pbar.update(1)
        self.pbar.set_postfix({"Epoch": self.epoch})
        return False  # Возвращаем False чтобы продолжить обучение

def parse_args():
    parser = argparse.ArgumentParser(description='Topic Modeling Pipeline with Configurable Parameters')

    # Text processing parameters
    parser.add_argument('--min_token_length', type=int, default=2,
                        help='Minimum length for tokens during preprocessing')

    # LDA model parameters
    parser.add_argument('--num_lda_topics', type=int, default=500,
                        help='Number of topics for LDA model')
    parser.add_argument('--lda_passes', type=int, default=15,
                        help='Number of passes for LDA training')
    parser.add_argument('--lda_workers', type=int, default=8,
                        help='Number of worker processes for LDA training')

    # Topic naming parameters
    parser.add_argument('--topic_words_for_naming', type=int, default=20,
                        help='Number of top words to use for topic naming')

    # Embedding parameters
    parser.add_argument('--sentence_model', type=str, default='all-MiniLM-L6-v2',
                        choices=['all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2',
                                 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1'],
                        help='Sentence Transformer model to use')
    parser.add_argument('--topic_embed_topn', type=int, default=10,
                        help='Number of top words to use for topic embeddings')

    # Topic assignment parameters
    parser.add_argument('--top_lda_topics_per_doc', type=int, default=10,
                        help='Number of LDA topics to assign per document')
    parser.add_argument('--top_semantic_topics_per_doc', type=int, default=5,
                        help='Number of semantic topics to assign per document')

    # Theme-word mapping parameters
    parser.add_argument('--theme_words_topk', type=int, default=20,
                        help='Number of words to match for predefined themes')

    # Similarity calculation parameters
    parser.add_argument('--topic_similarity_topk', type=int, default=20,
                        help='Number of similar words to track per topic')
    parser.add_argument('--similarity_power', type=int, default=2,
                        help='Exponent for similarity score calculation')

    # File parameters
    parser.add_argument('--input_file', type=str, default='simplified_lyrics.json',
                        help='Input JSON file with lyrics')
    parser.add_argument('--output_file', type=str, default='lyrics_meta_with_topics.json',
                        help='Output JSON file for processed data')
    parser.add_argument('--topics_info_file', type=str, default='topics_info.json',
                        help='Output file for topic information')
    parser.add_argument('--theme_words_file', type=str, default='theme_to_words.json',
                        help='Output file for theme-word mappings')

    return parser.parse_args()


args = parse_args()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
model = SentenceTransformer(args.sentence_model)


def preprocess(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if token.is_alpha
            and not token.is_stop
            and len(token.text) > args.min_token_length]


@lru_cache(maxsize=None)
def cached_get_topic_name(top_words_tuple):
    return get_topic_name(top_words_tuple)


def get_topic_embeddings(topic_names_dict, lda_model):
    topic_embeddings = {}
    for topic_id in topic_names_dict:
        top_words = lda_model.show_topic(topic_id, topn=args.topic_embed_topn)
        word_list = [word for word, _ in top_words]
        phrase = ' '.join(word_list)
        embedding = model.encode(phrase, convert_to_tensor=True)
        topic_embeddings[topic_id] = embedding
    return topic_embeddings


def get_document_embedding(text):
    return model.encode(text, convert_to_tensor=True)


def assign_topics(args_tuple):
    key, entry, doc_bow, topic_names, topic_scores, lda = args_tuple
    topic_distribution = lda.get_document_topics(doc_bow)
    top_topics = sorted(topic_distribution, key=lambda x: -x[1])[:args.top_lda_topics_per_doc]
    entry['lda_topics'] = [
        {
            'topic': t[0],
            'prob': float(t[1]),
            'name': topic_names[t[0]],
            'score': topic_scores[t[0]]
        } for t in top_topics
    ]
    return key, entry


def assign_topics_semantic(args_tuple):
    key, entry, topic_embeddings, topic_names = args_tuple
    lyrics = entry.get("lyrics_en", "")
    doc_embedding = get_document_embedding(lyrics)

    similarities = []
    for topic_id, topic_emb in topic_embeddings.items():
        similarity = util.cos_sim(doc_embedding, topic_emb).item()
        similarities.append((topic_id, similarity))

    top_topics = sorted(similarities, key=lambda x: -x[1])[:args.top_semantic_topics_per_doc]
    entry['semantic_topics'] = [
        {
            'topic': tid,
            'similarity': float(sim),
            'name': topic_names[tid]
        } for tid, sim in top_topics
    ]
    return key, entry


def find_top_words_for_themes(theme_names, dictionary):
    all_words = [dictionary[id] for id in dictionary.keys()]
    all_word_embeddings = model.encode(all_words, convert_to_tensor=True)

    theme_to_words = {}
    for theme in theme_names:
        theme_emb = model.encode(theme, convert_to_tensor=True)
        cos_sim = util.cos_sim(theme_emb, all_word_embeddings)[0]
        top_indices = torch.topk(cos_sim, args.theme_words_topk).indices.tolist()
        top_words = [all_words[i] for i in top_indices]
        top_scores = [cos_sim[i].item() for i in top_indices]
        theme_to_words[theme] = list(zip(top_words, top_scores))
    return theme_to_words


def main():
    print("Загрузка метаданных...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print("Предобработка текстов...")
    with multiprocessing.Pool() as pool:
        texts = list(tqdm(pool.imap(preprocess, (entry['lyrics_en'] for entry in meta.values())),
                          total=len(meta)))

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Обучение LDA...")
    with tqdm(total=args.lda_passes, desc="LDA Training") as pbar:
        lda = LdaMulticore(
            corpus,
            num_topics=args.num_lda_topics,
            id2word=dictionary,
            passes=args.lda_passes,
            workers=args.lda_workers,
            random_state=42,
            iterations=100,  # Увеличьте при необходимости
            eval_every=1,  # Оценка после каждой эпохи
        )

        # Вручную обновляем прогресс-бар
        for _ in range(args.lda_passes):
            lda.update(corpus)
            pbar.update(1)
            pbar.set_postfix({
                "Perplexity": f"{lda.log_perplexity(corpus):.3f}",
                "Topics": len(lda.get_topics())
            })
    print("Создание названий тем...")
    topic_names = {}
    topic_scores = {}
    for topic_id in tqdm(range(args.num_lda_topics)):
        top_words = lda.show_topic(topic_id, topn=args.topic_words_for_naming)
        name, score = cached_get_topic_name(tuple(top_words))
        topic_names[topic_id] = name
        topic_scores[topic_id] = score

    print("Создание информации о темах...")
    topics_info = []
    for topic_id in range(args.num_lda_topics):
        top_words = lda.show_topic(topic_id, topn=args.topic_words_for_naming)
        word_list = [word for word, _ in top_words]
        topics_info.append({
            "topic_id": topic_id,
            "name": topic_names[topic_id],
            "score": topic_scores[topic_id],
            "words": word_list
        })

    with open(args.topics_info_file, 'w', encoding='utf-8') as f:
        json.dump(topics_info, f, ensure_ascii=False, indent=2)

    print("Подбираем слова к заранее заданным темам...")
    candidate_names = load_candidate_names('candidate_names.txt')
    theme_to_words = find_top_words_for_themes(candidate_names, dictionary)
    with open(args.theme_words_file, 'w', encoding='utf-8') as f:
        json.dump(theme_to_words, f, ensure_ascii=False, indent=2)

    print("Генерация эмбеддингов слов из словаря...")
    all_dict_words = list(dictionary.values())
    word_embeddings = model.encode(all_dict_words, convert_to_tensor=True)

    print("Генерация эмбеддингов тем и поиск топ-слов...")
    topic_word_similarities = {}
    for topic_id in tqdm(range(args.num_lda_topics)):
        top_words = lda.show_topic(topic_id, topn=args.topic_words_for_naming)
        topic_word_list = [word for word, _ in top_words]
        topic_word_embs = model.encode(topic_word_list, convert_to_tensor=True)
        topic_embedding = torch.mean(topic_word_embs, dim=0, keepdim=True)

        cos_similarities = util.cos_sim(topic_embedding, word_embeddings)[0]
        top_indices = torch.topk(cos_similarities, k=args.topic_similarity_topk).indices.tolist()

        top_words_sim = []
        for idx in top_indices:
            word = all_dict_words[idx]
            similarity = cos_similarities[idx].item()
            top_words_sim.append([word, similarity])

        topic_name = topic_names[topic_id]
        topic_word_similarities[topic_name] = top_words_sim

    with open('topic_word_similarities.json', 'w', encoding='utf-8') as f:
        json.dump(topic_word_similarities, f, ensure_ascii=False, indent=2)

    print("Присвоение тем на основе распределения LDA...")
    lda_args = (
        (key, entry, doc_bow, topic_names, topic_scores, lda)
        for (key, entry), doc_bow in zip(meta.items(), corpus)
    )
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(assign_topics, lda_args), total=len(meta)))

    meta = dict(results)

    print("Генерация векторов тем...")
    topic_embeddings = get_topic_embeddings(topic_names, lda)

    print("Присвоение тем на основе векторного сходства...")
    semantic_args = (
        (key, entry, topic_embeddings, topic_names)
        for key, entry in meta.items()
    )
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(assign_topics_semantic, semantic_args), total=len(meta)))

    meta = dict(results)

    print("Присвоение тем на основе суммы квадратов близости слов...")


    def calc_topic_score_for_text(text_tokens, topic_words_sim):
        topic_word_dict = {w: sim for w, sim in topic_words_sim}
        score = 0.0
        for token in text_tokens:
            sim = topic_word_dict.get(token)
            if sim is not None:
                score += pow(sim, args.similarity_power)
        return score


    for key, entry in tqdm(meta.items()):
        text = entry.get('lyrics_en', '').lower()
        doc_tokens = preprocess(text)
        topic_scores_for_entry = []
        for topic_name, words_sim in topic_word_similarities.items():
            score = calc_topic_score_for_text(doc_tokens, words_sim)
            topic_scores_for_entry.append((topic_name, score))
        topic_scores_for_entry.sort(key=lambda x: -x[1])
        entry['topic_word_similarity_scores'] = [
            {"topic": t[0], "score": t[1]} for t in topic_scores_for_entry[:5]
        ]

    print("Сохранение финального файла...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Результаты сохранены в:\n"
          f"- Метаданные с темами: {args.output_file}\n"
          f"- Информация о темах: {args.topics_info_file}\n"
          f"- Сопоставление тем и слов: {args.theme_words_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()