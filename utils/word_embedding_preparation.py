# utils/embedding_processing.py

import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity as cosim
from itertools import groupby
from scipy.stats import pearsonr

def load_model(path):
    return Word2Vec.load(path)

def load_data(path, condition):
    df = pickle.load(open(path, "rb"))
    return df[df['hitscore'] > condition]

def generate_embeddings(model, vocab):
    def embedding(phrase):
        phrase = phrase.replace("*", "")
        sub = np.zeros(model.vector_size)
        try:
            return model.wv[phrase]
        except KeyError:
            words = phrase.replace('_', ' ').split()
            return sum((model.wv[word] for word in words if word in model.wv), np.zeros(model.vector_size))

    return np.array([embedding(word) for word in vocab])

def calculate_cosim_matrix(vocab_embed, lexicon_embed):
    return cosim(vocab_embed, lexicon_embed)

def calculate_pearson_matrix(embeddings):
    size = len(embeddings)
    return np.array([[pearsonr(embeddings[i], embeddings[j])[0] for j in range(size)] for i in range(size)])

def filter_embeddings(cosim_matrix, vocab, vocab_embed, threshold):
    filtered_vocab = [vocab[idx] for idx, val in enumerate(cosim_matrix) if max(val) > threshold]
    filtered_vembed = [vocab_embed[idx] for idx, val in enumerate(cosim_matrix) if max(val) > threshold]
    return filtered_vocab, filtered_vembed

# Example usage within this module (for module testing purposes only)
if __name__ == "__main__":
    model = load_model("/work/vedant/subreddit__300_word2vec.model")
    data = load_data("/work/vedant/Depression_score.pkl", 1)
    vocab = set(word for doc in data.ngrams for word in doc)
    vocab_embed = generate_embeddings(model, vocab)
    lexicon_embed = generate_embeddings(model, ["example", "test"])
    cosim_matrix = calculate_cosim_matrix(vocab_embed, lexicon_embed)
    print(cosim_matrix)
