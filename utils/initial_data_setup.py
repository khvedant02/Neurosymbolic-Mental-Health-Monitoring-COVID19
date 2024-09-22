# utils/data_preparation.py

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def load_stopwords(filepath):
    with open(filepath, 'r') as file:
        stop_string = file.read()
    return set(stop_string.split())

def clean_text(text, stop_words, common_terms):
    text = re.sub(r'http\S+', '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in common_terms
                       and word not in string.punctuation and len(word) > 2 and '.' not in word and "'" not in word]
    return " ".join(filtered_tokens)

def preprocess_texts(corpus, stop_filepath):
    common_terms = load_stopwords(stop_filepath)
    stop_words = set(stopwords.words('english'))
    return [word_tokenize(clean_text(doc, stop_words, common_terms)) for doc in corpus]
