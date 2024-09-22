# model_training.py

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

def train_ngram_models(corpus):
    bigram = Phrases(corpus, min_count=1, threshold=0.75)
    trigram = Phrases(bigram[corpus], min_count=1, threshold=0.5)
    return Phraser(bigram), Phraser(trigram)

def extract_ngrams(bigram_model, trigram_model, corpus):
    return [trigram_model[bigram_model[doc]] for doc in corpus]

def train_word2vec(sentences, vector_size=300, window=5, min_count=1, workers=32):
    return Word2Vec(sentences=sentences, size=vector_size, window=window, min_count=min_count, workers=workers)
