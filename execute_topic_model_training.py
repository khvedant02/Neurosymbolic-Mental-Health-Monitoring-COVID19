# run_training.py

import os
import pandas as pd
from utils.initial_data_setup import preprocess_texts
from topic_model_training import train_ngram_models, apply_ngram_models, train_lda_model
import pickle

def load_data(file_path):
    return pd.read_pickle(file_path, encoding='latin-1')

def main():
    data_paths = ['/path/to/data']
    stop_words_path = '/path/to/terrier-stop.txt'
    output_dir = '/path/to/output'
    num_topics = 90
    chunksize = 2000
    passes = 20
    iterations = 100

    for data_path in data_paths:
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pickl')]
        sentences = []
        for file in files:
            df = load_data(file)
            sentences.extend(df['text'].dropna().tolist())
        corpus = preprocess_texts(sentences, stop_words_path)
        
        bigram_model, trigram_model = train_ngram_models(corpus)
        ngrams = apply_ngram_models(bigram_model, trigram_model, corpus)

        dictionary, lda_model = train_lda_model(ngrams, num_topics, chunksize, passes, iterations)
        lda_model.save(os.path.join(output_dir, 'lda_model.model'))
        dictionary.save(os.path.join(output_dir, 'lda_dictionary.pkl'))

        # Save ngram models
        with open(os.path.join(output_dir, 'bigram_model.pkl'), 'wb') as f:
            pickle.dump(bigram_model, f)
        with open(os.path.join(output_dir, 'trigram_model.pkl'), 'wb') as f:
            pickle.dump(trigram_model, f)

if __name__ == '__main__':
    main()
