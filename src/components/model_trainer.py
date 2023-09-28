import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.components.logger import logging
from src.components.exception import CustomException
from src.components.utils import save_file, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])
            # dictionary_of_models

            models = {"Random Forest": RandomForestRegressor(),
                      "Decision Tree": DecisionTreeRegressor(),
                      "Gradient Boosting": GradientBoostingRegressor(),
                      "Linear Regression": LinearRegression(),
                      "KNeighbors Classifier": KNeighborsRegressor(),
                      "XGBClassifier": XGBRegressor(),
                      "CatBoostingClassifier": CatBoostRegressor(verbose=False),
                      "AdaBoostClassifier": AdaBoostRegressor()
                      }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  # to_refine_nested_list

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("no best model found")
            logging.info(f"best model found on training and testing dataset")

            save_file(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model)  # best model is pickled

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square, best_model_name
        except Exception as e:
            raise CustomException(e, sys)
