import json
import spacy
import torch
from gensim import corpora
from gensim.models import LdaMulticore
from tqdm import tqdm
import multiprocessing
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

from topics_add.lda_topic_namer_with_embeddings import get_topic_name  # <- возвращает (name, score)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 2]

@lru_cache(maxsize=None)
def cached_get_topic_name(top_words_tuple):
    return get_topic_name(top_words_tuple)

def get_topic_embeddings(topic_names_dict, lda_model, topn=10):
    topic_embeddings = {}
    for topic_id in topic_names_dict:
        top_words = lda_model.show_topic(topic_id, topn=topn)
        word_list = [word for word, _ in top_words]
        phrase = ' '.join(word_list)
        embedding = model.encode(phrase, convert_to_tensor=True)
        topic_embeddings[topic_id] = embedding
    return topic_embeddings

def get_document_embedding(text):
    return model.encode(text, convert_to_tensor=True)

def assign_topics(args):
    key, entry, doc_bow, topic_names, topic_scores, lda = args
    topic_distribution = lda.get_document_topics(doc_bow)
    top_topics = sorted(topic_distribution, key=lambda x: -x[1])[:10]
    entry['lda_topics'] = [
        {
            'topic': t[0],
            'prob': float(t[1]),
            'name': topic_names[t[0]],
            'score': topic_scores[t[0]]
        } for t in top_topics
    ]
    return key, entry

def assign_topics_semantic(args):
    key, entry, topic_embeddings, topic_names = args
    lyrics = entry.get("lyrics_en", "")
    doc_embedding = get_document_embedding(lyrics)

    similarities = []
    for topic_id, topic_emb in topic_embeddings.items():
        similarity = util.cos_sim(doc_embedding, topic_emb).item()
        similarities.append((topic_id, similarity))

    top_topics = sorted(similarities, key=lambda x: -x[1])[:5]
    entry['semantic_topics'] = [
        {
            'topic': tid,
            'similarity': float(sim),
            'name': topic_names[tid]
        } for tid, sim in top_topics
    ]
    return key, entry

def main():
    input_filename = 'simplified_lyrics.json'
    output_filename = 'lyrics_meta_with_topics.json'
    topics_info_filename = 'topics_info.json'
    num_topics = 500

    print("Загрузка метаданных...")
    with open(input_filename, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print("Предобработка текстов...")
    with multiprocessing.Pool() as pool:
        texts = list(tqdm(pool.imap(preprocess, (entry['lyrics_en'] for entry in meta.values())), total=len(meta)))

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Обучение LDA...")
    lda = LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15,
        workers=8,
        random_state=42
    )

    print("Создание названий тем...")
    topic_names = {}
    topic_scores = {}
    for topic_id in tqdm(range(num_topics)):
        top_words = lda.show_topic(topic_id, topn=20)
        name, score = cached_get_topic_name(tuple(top_words))  # tuple для кэша
        topic_names[topic_id] = name
        topic_scores[topic_id] = score

    print("Создание информации о темах...")
    topics_info = []
    for topic_id in range(num_topics):
        top_words = lda.show_topic(topic_id, topn=20)
        word_list = [word for word, _ in top_words]
        topics_info.append({
            "topic_id": topic_id,
            "name": topic_names[topic_id],
            "score": topic_scores[topic_id],
            "words": word_list
        })

    with open(topics_info_filename, 'w', encoding='utf-8') as f:
        json.dump(topics_info, f, ensure_ascii=False, indent=2)


    # === Новый блок: генерация эмбеддингов слов словаря и топ-20 слов по близости к темам ===

    print("Генерация эмбеддингов слов из словаря...")
    all_dict_words = list(dictionary.values())
    word_embeddings = model.encode(all_dict_words, convert_to_tensor=True)

    print("Генерация эмбеддингов тем и поиск топ-20 слов по близости...")
    topic_word_similarities = {}

    for topic_id in tqdm(range(num_topics)):
        top_words = lda.show_topic(topic_id, topn=20)
        topic_word_list = [word for word, _ in top_words]
        topic_word_embs = model.encode(topic_word_list, convert_to_tensor=True)
        topic_embedding = torch.mean(topic_word_embs, dim=0, keepdim=True)

        cos_similarities = util.cos_sim(topic_embedding, word_embeddings)[0]
        top_indices = torch.topk(cos_similarities, k=20).indices.tolist()

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


    # === Новый блок: присвоение тем по сумме квадратов близости слов ===

    print("Присвоение тем на основе суммы квадратов близости слов...")

    def calc_topic_score_for_text(text_tokens, topic_words_sim):
        topic_word_dict = {w: sim for w, sim in topic_words_sim}
        score = 0.0
        for token in text_tokens:
            sim = topic_word_dict.get(token)
            if sim is not None:
                score += sim * sim * sim
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
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Файл с темами сохранён как: {output_filename}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
