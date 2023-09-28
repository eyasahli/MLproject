import os
import sys
from src.components.exception import CustomException
from src.components.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("started data ingestion component or method")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),
                        exist_ok=True)  # creating the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # dropping data there

            logging.info("train test sets initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,
                             header=True)  # dropping training data there
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # dropping test data there

            logging.info("data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr, test_arr))
