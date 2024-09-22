# run_training.py

import os
import pickle
from utils.text_cleaning import load_stopwords, preprocess_texts
from nlp_model_training import train_ngram_models, extract_ngrams, train_word2vec

def load_data(files):
    """Load and aggregate text data from multiple pickle files."""
    all_texts = []
    for file in files:
        with open(file, 'rb') as f:
            df = pickle.load(f, encoding='latin-1')
            all_texts.extend(df['text'].dropna().tolist())
    return all_texts

def main():
    fpath = '/work/vedant/Others/subreddit/'
    categories = ['depression/', 'anxiety/', 'addiction/']
    filelist = [os.path.join(fpath, cat, fname) for cat in categories for fname in os.listdir(os.path.join(fpath, cat)) if fname.endswith(".pickl")]

    stop_words = load_stopwords('/work/vedant/Others/terrier-stop.txt')
    texts = load_data(filelist)
    processed_texts = preprocess_texts(texts, stop_words)
    
    bigram_model, trigram_model = train_ngram_models(processed_texts)
    ngrams = extract_ngrams(bigram_model, trigram_model, processed_texts)
    
    word2vec_model = train_word2vec(ngrams)
    word2vec_model.save("/work/vedant/subreddit_word2vec.model")
    print('Model training completed and saved.')

if __name__ == '__main__':
    main()
