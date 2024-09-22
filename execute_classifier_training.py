from utils.train_test_split import load_model, load_and_preprocess_data
from machine_learning_model_training import train_models
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def main():
    model_path = "/work/vedant/subreddit__300_word2vec.model"
    data_path = "/work/vedant/Addiction_score.pkl"
    weight_path = "/work/vedant/W.pkl"
    model = load_model(model_path)
    X, y = load_and_preprocess_data(model, data_path, weight_path, 'Addiction')

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'GaussianNB': GaussianNB(),
        'Subsampled Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True, max_samples=0.5),
        'Balanced Subsample Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    }

    results = train_models(X, y, classifiers)
    for classifier_name, metrics in results.items():
        print(f"{classifier_name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save the final results
    with open("/work/vedant/training_results.pkl", "wb") as f:
        pkl.dump(results, f)

if __name__ == "__main__":
    main()
