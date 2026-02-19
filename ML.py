from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    print("----- SVM Results -----")
    print("Accuracy:", accuracy_score(y_test, svm_predictions))
    print(classification_report(y_test, svm_predictions))

    # 4. Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)

    print("----- KNN Results -----")
    print("Accuracy:", accuracy_score(y_test, knn_predictions))
    print(classification_report(y_test, knn_predictions))


if __name__ == "__main__":
    main()
