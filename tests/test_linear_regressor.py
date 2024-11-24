import pytest
import numpy as np
import pandas as pd
from dlpml.regression.linear_regressor import LinearRegressor

class TestLinearRegressor:

    @pytest.fixture
    def data(self):
        column_names = ["Var1", "Var2"]
        data = pd.read_csv("data/ex_linear_regression_data1.csv", header=None, names=column_names)
        X_train = data.iloc[:, [0]].to_numpy() 
        y_train = data.iloc[:, 1].to_numpy()
        return X_train, y_train

    @pytest.fixture
    def model(self):
        return LinearRegressor(alpha=0.01, iterations=10*1_000, lambda_=0.01)

    def test_initialization(self, model):
        assert model.alpha == 0.01
        assert model.iterations == 10*1_000
        assert model.lambda_ == 0.01

    def test_fit(self, model, data):
        X_train, y_train = data
        X_train_scaled = (X_train - np.mean(X_train, axis=0) - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
        y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)
        model.fit(X_train_scaled, y_train_scaled)
        assert model.w is not None
        assert model.b is not None

    def test_predict(self, model, data):
        X_train, y_train = data
        X_train_scaled = (X_train - np.mean(X_train, axis=0) - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
        y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)
        model.fit(X_train_scaled, y_train_scaled)
        y_pred = model.predict(X_train_scaled)
        assert y_pred.shape == y_train_scaled.shape
