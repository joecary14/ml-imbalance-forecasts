import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import misc.plotting as plotting
import misc.constants as ct

def run_regression_model(X: pd.DataFrame,
                          y: pd.Series,
                          model_type: str,
                          test_size: float,
                          **model_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    pipeline = train_regression_model(X_train, y_train, model_type, **model_params)
    metrics = evaluate_regression_model(pipeline, X_test, y_test)
    print(f"Metrics: {metrics}")
    plotting.plot_predictions_vs_actuals(y_test, pipeline.predict(X_test))

def train_regression_model(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           model_type: str,
                           **model_params):
    if model_type == ct.ModelTypes.RANDOM_FOREST.value:
        regressor = RandomForestRegressor(**model_params)
    elif model_type == ct.ModelTypes.LINEAR_REGRESSION.value:
        regressor = LinearRegression(**model_params)
    elif model_type == ct.ModelTypes.QUANTILE_REGRESSION.value:
        regressor = QuantileRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'mse': mse, 'rmse': rmse, 'r2': r2}
    return metrics