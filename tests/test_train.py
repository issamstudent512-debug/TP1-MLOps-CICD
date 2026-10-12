"""
Tests unitaires pour le script d'entrainement.
"""

import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from train import load_data, train_model, evaluate_model


def test_load_data_shapes():
    """Verifie que les donnees sont correctement decoupees."""
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_model_returns_fitted_model():
    """Verifie que le modele entraine peut faire des predictions."""
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_evaluate_model_returns_valid_accuracy():
    """Verifie que l'accuracy est bien comprise entre 0 et 1."""
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0
