# utils/training_data_procesing.py
import pandas as pd
import numpy as np
import pickle as pkl
from gensim.models import Word2Vec

def load_model(path):
    return Word2Vec.load(path)

def load_weights(path):
    return pkl.load(open(path, "rb"))

def embedding(model, phrase, weights):
    sub = np.zeros(model.vector_size)
    words = phrase.replace('*', '').replace('_', ' ').split()
    for word in words:
        if word in model.wv:
            weighted_embedding = model.wv[word]
            if word in weights:
                weighted_embedding *= weights[word]
            sub += weighted_embedding
    return sub

def modulate_embeddings(model, phrases, category_weights):
    return np.array([embedding(model, phrase, category_weights) for phrase in phrases])

def load_and_preprocess_data(model, data_path, weight_path, category):
    data = pkl.load(open(data_path, "rb"))
    weights = load_weights(weight_path)
    category_weights = weights[category]
    data['mod_embeddings'] = modulate_embeddings(model, data['ngrams'], category_weights)
    return data['mod_embeddings'], data['label'].values
