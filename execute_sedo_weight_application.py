from utils.word_embedding_preparation import load_model, load_data, generate_embeddings, calculate_cosim_matrix, filter_embeddings
from utils.sedo_weight_calculation import train_sedo
import pickle

def main():
    model_path = "/work/vedant/subreddit__300_word2vec.model"
    data_path = "/work/vedant/Depression_score.pkl"
    model = load_model(model_path)
    data = load_data(data_path, 1)
    vocab = set(word for doc in data.ngrams for word in doc)
    vocab_embed = generate_embeddings(model, vocab)
    lexicon = ["example", "test"]  # Define your lexicon
    lexicon_embed = generate_embeddings(model, lexicon)
    cosim_matrix = calculate_cosim_matrix(vocab_embed, lexicon_embed)
    filtered_vocab, filtered_vembed = filter_embeddings(cosim_matrix, vocab, vocab_embed, 0.9)
    W = train_sedo(filtered_vembed, filtered_vembed, filtered_vembed)  # Dummy placeholders for actual data
    pickle.dump(W, open("/work/vedant/FinalWmatrix.pkl", "wb"))

if __name__ == '__main__':
    main()
