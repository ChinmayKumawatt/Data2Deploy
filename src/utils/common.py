import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params, threshold_config):
    try:
        report = {}

        for model_name, model in models.items():
            param_dist = params.get(model_name, {})

            rs = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                n_jobs=-1,
                verbose=0
            )

            rs.fit(X_train, y_train)

            best_model = rs.best_estimator_

            best_threshold = 0.5
            best_f1 = -1

            if threshold_config.get(model_name, False) and hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(X_test)[:, 1]
                thresholds = np.linspace(0.2, 0.8, 25)

                for t in thresholds:
                    preds = (probs >= t).astype(int)
                    score = f1_score(y_test, preds)

                    if score > best_f1:
                        best_f1 = score
                        best_threshold = t

                y_test_pred = (probs >= best_threshold).astype(int)
                y_train_pred = (best_model.predict_proba(X_train)[:, 1] >= best_threshold).astype(int)

            else:
                y_test_pred = best_model.predict(X_test)
                y_train_pred = best_model.predict(X_train)

            report[model_name] = {
                "model": best_model,
                "best_params": rs.best_params_,
                "best_threshold": best_threshold,
                "metrics": {
                    "train_f1": f1_score(y_train, y_train_pred),
                    "test_f1": f1_score(y_test, y_test_pred),
                    "precision": precision_score(y_test, y_test_pred),
                    "recall": recall_score(y_test, y_test_pred)
                }
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
