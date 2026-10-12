"""
Script d'entrainement d'un modele de classification sur le jeu de donnees Iris.
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """Charge le jeu depython -m ensurepip --upgrade donnees Iris et le divise en train/test."""
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraine un arbre de decision sur les donnees d'entrainement."""
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Retourne l'accuracy du modele sur le jeu de test."""
    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Accuracy du modele : {accuracy:.4f}")

    # Sauvegarde du modele entraine
    joblib.dump(model, "model.pkl")
    print("Modele sauvegarde dans model.pkl")

    # Seuil minimal de qualite : le pipeline doit echouer si non atteint
    SEUIL_MINIMAL = 0.85

    if accuracy < SEUIL_MINIMAL:
        raise ValueError(
            f"Accuracy {accuracy:.4f} inferieure au seuil minimal "
            f"{SEUIL_MINIMAL}"
        )
