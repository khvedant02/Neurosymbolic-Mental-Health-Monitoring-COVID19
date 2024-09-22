# utils/text_preprocessing.py

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def load_stopwords(filepath):
    with open(filepath, 'r') as file:
        stop_words = file.read().split()
    return set(stopwords.words('english')).union(set(stop_words))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    tokens = word_tokenize(text.lower())
    return tokens

def filter_tokens(tokens, stop_words, min_length=3):
    """Remove stopwords and tokens shorter than min_length, excluding punctuation."""
    return [token for token in tokens if token not in stop_words and token not in string.punctuation and len(token) >= min_length]

def preprocess_texts(corpus, stop_words):
    return [filter_tokens(clean_text(doc), stop_words) for doc in corpus]
