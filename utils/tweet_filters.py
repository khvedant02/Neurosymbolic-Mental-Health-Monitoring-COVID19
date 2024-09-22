# utils/tweet_filters.py

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def intersection(lst1, lst2):
    """Compute the intersection of two lists."""
    return list(set(lst1) & set(lst2))

def locationfilter(allnameslist, tweet):
    """Filter tweets based on whether they mention any of the names in the allnameslist."""
    words = set(nltk.word_tokenize(tweet.lower()))
    return 1 if not intersection(allnameslist, words) else 0

def categoryfilter(category_lexicon, tweet):
    """Filter tweets based on whether they contain any keywords related to specific categories."""
    words = set(nltk.word_tokenize(tweet.lower()))
    return 1 if intersection(category_lexicon, words) else 0

def languagefilter(tweet_object):
    """Filter tweets that are not in English."""
    return 1 if tweet_object.get('lang') == 'en' else 0
