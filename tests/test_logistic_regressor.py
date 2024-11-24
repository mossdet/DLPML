import pytest
import numpy as np
from dlpml.classification.logistic_regressor import LogisticRegressor

class TestLogisticRegressor:

    def test_initialization(self):
        model = LogisticRegressor(alpha=0.01, iterations=1000, lambda_=0.01)
        assert model.alpha == 0.01
        assert model.iterations == 1000
        assert model.lambda_ == 0.01

    def test_sigmoid(self):
        model = LogisticRegressor()
        z = np.array([0, 2])
        expected_output = np.array([0.5, 0.88079708])
        np.testing.assert_almost_equal(model.sigmoid(z), expected_output, decimal=5)

    def test_gradient_descent(self):
        model = LogisticRegressor(alpha=0.01, iterations=100)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        w_in = np.array([0.1, 0.2])
        b_in = 0.3
        w, b = model.gradient_descent(X, y, w_in, b_in)
        assert w.shape == w_in.shape
        assert isinstance(b, float)

    def test_fit(self):
        model = LogisticRegressor(alpha=0.01, iterations=100)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(X, y)
        assert model.w is not None
        assert model.b is not None

    def test_predict(self):
        model = LogisticRegressor()
        model.w = np.array([0.1, 0.2])
        model.b = 0.3
        X = np.array([[1, 2], [3, 4]])
        predictions = model.predict(X)
        assert predictions.shape == (2,)
        assert np.array_equal(predictions, np.array([1, 1]))