import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

def train_models(X, y, classifiers):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5)
    results = {}

    for name, clf in classifiers.items():
        accuracies, precisions, recalls, f1scores, aucs = [], [], [], [], []
        for train_index, test_index in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_probas = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_probas)
            precisions.append(pr)
            recalls.append(rc)
            f1scores.append(f1)
            aucs.append(auc_score)

        results[name] = {
            'Accuracy': np.mean(accuracies),
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1 Score': np.mean(f1scores),
            'AUC': np.mean(aucs)
        }
    return results
