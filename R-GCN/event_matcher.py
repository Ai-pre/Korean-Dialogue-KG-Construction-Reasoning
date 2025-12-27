# event_matcher_tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class EventMatcher:
    def __init__(self, node_texts):
        self.node_texts = node_texts
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000
        )
        self.node_vecs = self.vectorizer.fit_transform(node_texts)

    def find_topk(self, query, k=5):
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.node_vecs.T).toarray()[0]
        topk_idx = np.argsort(scores)[::-1][:k]
        return topk_idx.tolist(), scores[topk_idx].tolist()
