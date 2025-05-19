from sentence_transformers import SentenceTransformer, util
import torch

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config  # Сохраняем конфигурацию в атрибут объекта
        self.model = SentenceTransformer(config['embedding']['sentence_model'])

    def get_topic_embeddings(self, topic_names, lda_model):
        topic_embeddings = {}
        for topic_id in topic_names:
            # Используем self.config для доступа к параметрам
            top_words = lda_model.show_topic(topic_id, self.config['topic_naming']['topic_embed_topn'])
            phrase = ' '.join([word for word, _ in top_words])
            embedding = self.model.encode(phrase, convert_to_tensor=True)
            topic_embeddings[topic_id] = embedding
        return topic_embeddings

    def get_document_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)