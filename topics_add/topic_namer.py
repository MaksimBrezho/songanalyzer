from functools import lru_cache
from tqdm import tqdm
from lda_topic_namer_with_embeddings import get_topic_name
from sentence_transformers import util
import torch

class TopicNamer:
    def __init__(self, config, model):
        self.config = config
        self.model = model  # SentenceTransformer instance

    @lru_cache(maxsize=None)
    def cached_get_topic_name(self, top_words_tuple):
        return get_topic_name(top_words_tuple)

    def name_topics(self, lda_model, progress_cb=None):
        topic_names = {}
        topic_scores = {}
        for topic_id in range(lda_model.num_topics):
            top_words = lda_model.show_topic(topic_id, self.config['topic_naming']['topic_words_for_naming'])
            name, score = self.cached_get_topic_name(tuple(top_words))
            topic_names[topic_id] = name
            topic_scores[topic_id] = score
            if progress_cb:
                progress_cb.update_task(1)
        return topic_names, topic_scores

    def find_top_words_for_themes(self, theme_names, dictionary):
        all_words = [dictionary[id] for id in dictionary.keys()]
        all_word_embeddings = self.model.encode(all_words, convert_to_tensor=True)

        theme_to_words = {}
        for theme in theme_names:
            theme_emb = self.model.encode(theme, convert_to_tensor=True)
            cos_sim = util.cos_sim(theme_emb, all_word_embeddings)[0]
            top_indices = torch.topk(cos_sim, self.config['theme_mapping']['theme_words_topk']).indices.tolist()
            top_words = [all_words[i] for i in top_indices]
            top_scores = [cos_sim[i].item() for i in top_indices]
            theme_to_words[theme] = list(zip(top_words, top_scores))
        return theme_to_words
