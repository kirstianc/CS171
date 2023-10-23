import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def main():
    # Load and prep data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Support Vector Machine
    svm_model = SVC(probability=True)
    svm_model.fit(X, y)
    svm_scores = cross_val_score(svm_model, X, y, cv=5)
    svm_predictions = svm_model.predict(X)
    svm_prob = svm_model.predict_proba(X)
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    rf_scores = cross_val_score(rf_model, X, y, cv=5)
    rf_predictions = rf_model.predict(X)
    rf_prob = rf_model.predict_proba(X)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X, y)
    knn_scores = cross_val_score(knn_model, X, y, cv=5)
    knn_predictions = knn_model.predict(X)
    knn_prob = knn_model.predict_proba(X)

    # Print the evaluation metrics
    print('SVM Metrics ----')
    print('SVM Accuracy: {:.2f}%'.format(svm_scores.mean() * 100))
    print('SVM F1 Score: {:.2f}%'.format(f1_score(y, svm_predictions, average='macro') * 100))
    print('SVM ROC-AUC Score: {:.2f}%'.format(roc_auc_score(y, svm_prob, multi_class='ovr') * 100))
    
    print('\nRandom Forest Metrics ----')
    print('Random Forest Accuracy: {:.2f}%'.format(rf_scores.mean() * 100))
    print('Random Forest F1 Score: {:.2f}%'.format(f1_score(y, rf_predictions, average='macro') * 100))
    print('Random Forest ROC-AUC Score: {:.2f}%'.format(roc_auc_score(y, rf_prob, multi_class='ovr') * 100))
    
    print('\nKNN Metrics ----')
    print('KNN Accuracy: {:.2f}%'.format(knn_scores.mean() * 100))
    print('KNN F1 Score: {:.2f}%'.format(f1_score(y, knn_predictions, average='macro') * 100))
    print('KNN ROC-AUC Score: {:.2f}%'.format(roc_auc_score(y, knn_prob, multi_class='ovr') * 100))

if __name__ == '__main__':
    main()