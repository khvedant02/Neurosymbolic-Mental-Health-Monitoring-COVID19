# model_training.py

from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamulticore import LdaMulticore
import gensim
import pickle

def train_ngram_models(corpus):
    bigram = Phrases(corpus, min_count=1, threshold=0.75)
    trigram = Phrases(bigram[corpus], min_count=1, threshold=0.5)
    return Phraser(bigram), Phraser(trigram)

def apply_ngram_models(bigram_model, trigram_model, corpus):
    return [trigram_model[bigram_model[doc]] for doc in corpus]

def train_lda_model(ngrams, num_topics, chunksize, passes, iterations):
    dictionary = gensim.corpora.Dictionary(ngrams)
    corpus = [dictionary.doc2bow(doc) for doc in ngrams]
    model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary.id2token,
        chunksize=chunksize,
        iterations=iterations,
        num_topics=num_topics,
        workers=20,
        passes=passes
    )
    return dictionary, model
