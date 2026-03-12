"""Tests for phishdet.model (train / save / load / predict)."""

import os
import tempfile

import numpy as np
import pytest

from phishdet.model import load_model, predict, predict_proba, save_model, train

# ------------------------------------------------------------------ fixtures

LEGIT_URLS = [
    "https://www.google.com",
    "https://www.github.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.apple.com",
]
PHISH_URLS = [
    "http://192.168.1.1/login/secure",
    "http://paypal.com-secure-login.phishing.com/verify",
    "http://www.amazon-security-alert.com/login",
    "http://login-secure.bankofamerica.evil.com/account",
    "http://update-your-account.net/paypal/signin",
]

ALL_URLS = LEGIT_URLS + PHISH_URLS
ALL_LABELS = [0] * len(LEGIT_URLS) + [1] * len(PHISH_URLS)


@pytest.fixture(scope="module")
def trained_clf():
    """Return a fitted classifier trained on the small fixture dataset."""
    return train(ALL_URLS, ALL_LABELS, n_estimators=10, random_state=0)


# ------------------------------------------------------------------ tests

class TestTrain:
    def test_returns_classifier(self, trained_clf):
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(trained_clf, RandomForestClassifier)

    def test_classifier_is_fitted(self, trained_clf):
        # Fitted classifiers expose n_classes_
        assert hasattr(trained_clf, "classes_")
        assert list(trained_clf.classes_) == [0, 1]


class TestSaveLoad:
    def test_round_trip(self, trained_clf):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            save_model(trained_clf, model_path)
            assert os.path.exists(model_path)

            loaded = load_model(model_path)
            # Predictions must be identical after reload
            preds_original = predict(trained_clf, ALL_URLS)
            preds_loaded = predict(loaded, ALL_URLS)
            np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")


class TestPredict:
    def test_predict_returns_array(self, trained_clf):
        preds = predict(trained_clf, ALL_URLS)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(ALL_URLS),)

    def test_predict_values_binary(self, trained_clf):
        preds = predict(trained_clf, ALL_URLS)
        assert set(preds).issubset({0, 1})

    def test_predict_single_string(self, trained_clf):
        preds = predict(trained_clf, "https://www.google.com")
        assert preds.shape == (1,)

    def test_predict_proba_shape(self, trained_clf):
        probas = predict_proba(trained_clf, ALL_URLS)
        assert probas.shape == (len(ALL_URLS), 2)

    def test_predict_proba_sums_to_one(self, trained_clf):
        probas = predict_proba(trained_clf, ALL_URLS)
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(len(ALL_URLS)), atol=1e-6)
