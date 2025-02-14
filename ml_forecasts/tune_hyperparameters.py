import random
import math
import numpy as np
import pandas as pd
import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization

def get_best_hyperparameters(tuner, X_train, y_train, epochs, validation_split):
    tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_hyperparameters
    
def bayesian_optimization_tuning_model(build_model_fn, hyperparameter_bounds: dict,
                                       X: pd.DataFrame, y: pd.Series,
                                       test_split: float,
                                       window_size: int,
                                       batch_size: int,
                                       epochs: int,
                                       validation_split: float,
                                       init_points: int = 5,
                                       n_iter: int = 25):
    
    preprocessing.check_sufficient_data(X, batch_size, epochs)
    
    X_scaled = preprocessing.scale_independent_variables(X)
    y_scaled, dependent_variable_scaler = preprocessing.scale_dependent_variable(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_split, shuffle=False)
    
    X_train_seq, y_train_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    X_test_seq, y_test_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    # Define the objective function for Bayesian optimization.
    # It must take hyperparameters as keyword arguments.
    # We also convert any hyperparameters to integers if necessary.
    def objective(**hp):
        for param in hp:
            if isinstance(hyperparameter_bounds[param][0], int):
                hp[param] = int(round(hp[param]))
        print("Evaluating with hyperparameters:", hp)
        
        model = build_model_fn(input_shape, **hp)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_seq, y_train_seq,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        
        loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print("Loss:", loss)
        return -loss  # Negative because we want to minimize loss (optimizer maximizes)

    # Initialize the Bayesian optimizer.
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=hyperparameter_bounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_hp = optimizer.max['params']
    for param in best_hp:
        if isinstance(hyperparameter_bounds[param][0], int):
            best_hp[param] = int(round(best_hp[param]))

    best_model = build_model_fn(input_shape, **best_hp)
    best_model.fit(
        X_train_seq, y_train_seq,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    best_metric = best_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print("Best hyperparameters:", best_hp)
    print("Best test loss:", best_metric)

    y_pred_test = best_model.predict(X_test_seq)
    y_pred_test_unscaled = dependent_variable_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_unscaled = dependent_variable_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
    print("Test R^2:", r2)
    plotting.plot_predictions_vs_actuals(y_pred_test_unscaled, y_test_unscaled)

    return best_hp, best_model, best_metric


def random_search_tuning_model(build_model_fn, hyperparameter_space: dict,
                               X: pd.DataFrame, y: pd.Series,
                               test_split: float,
                               window_size: int,
                               max_iterations: int,
                               batch_size: int,
                               epochs: int,
                               validation_split: float):
    
    preprocessing.check_sufficient_data(X, batch_size, epochs)
    
    X_scaled = preprocessing.scale_independent_variables(X)
    y_scaled, dependent_variable_scaler = preprocessing.scale_dependent_variable(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_split, shuffle=False)

    X_train_seq, y_train_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    X_test_seq, y_test_seq = preprocessing.create_sequences(X_test, y_test, window_size)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    
    best_metric = np.inf
    best_hp = None
    best_model = None

    for i in range(max_iterations):
        hp = {}
        for param, bounds in hyperparameter_space.items():
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                hp[param] = random.randint(bounds[0], bounds[1])
            else:
                hp[param] = random.uniform(bounds[0], bounds[1])
        print(f"Iteration {i+1} with hyperparameters: {hp}")
        
        model = build_model_fn(input_shape, **hp)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train_seq, y_train_seq,
                  validation_split=validation_split,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[early_stopping],
                  verbose=1)
        
        loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"Iteration {i+1} test loss: {loss}")
        
        if loss < best_metric:
            best_metric = loss
            best_hp = hp
            best_model = model

    print("Best hyperparameters:", best_hp)
    print("Best test loss:", best_metric)
    
    x_test_seq, y_test_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    y_pred_test = best_model.predict(x_test_seq)
    
    y_pred_test_unscaled = dependent_variable_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_unscaled = dependent_variable_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
    print("Test R^2:", r2)
    
    plotting.plot_predictions_vs_actuals(y_pred_test_unscaled, y_test_unscaled)
    
    return best_hp, best_model, best_metric