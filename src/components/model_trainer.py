import os
import sys 
from dataclasses import dataclass 
from src.utils import save_object,evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test data")
            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )    

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),  
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
    "Decision Tree": {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    "Random Forest": {
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'criterion': ['mse', 'mae'],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    "Gradient Boosting": {
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    "Linear Regression": {},
    "K-neighbors Regressor": {
        'n_neighbors': [5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "XGBRegressor": {
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "CatBoosting Regressor": {
        'iterations': [30, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10]
    },
    "AdaBoost Regressor": {
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'loss': ['linear', 'square', 'exponential']
    }
}


            model_report:dict = evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,param=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test,predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)

