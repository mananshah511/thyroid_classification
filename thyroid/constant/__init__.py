import os
from datetime import datetime

ROOT_DIR = os.getcwd()
CONFIF_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIF_DIR,CONFIG_FILE_NAME)


CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

DATABASE_NAME = "thyroid_classification.csv"

COLUMN_KEY = "columns"
NUMERIC_COULMN_KEY = "numerical_columns"
CATEGORICAL_COLUMN_KEY = "categorical_columns"
TARGET_COLUMN_KEY = "target_column"

NO_CLUSTER = 5

DROP_COLUMN_LIST = ['TBG','TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','FTI']


#training pipeline related variables

TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"

#data ingestion related variables

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR = "ingested_dir"
DATA_INGESTION_INGESTED_TRAIN_DATA_DIR = "ingested_train_dir"
DATA_INGESTION_INGESTED_TEST_DATA_DIR = "ingested_test_dir"

#data validation related variables

DATA_VALIDTION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_KEY = "schema_file"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME = "report_page_file_name"

#data transform related varibales

DATA_TRANSFORM_CONFIG_KEY = "data_transform_config"
DATA_TRANSFORM_DIR = "data_transform"
DATA_TRANSFORM_GRAPH_DIR_KEY = "graph_save_dir"
DATA_TRANSFORM_TRAIN_DIR_KEY = "train_dir"
DATA_TRANSFORM_TEST_DIR_KEY = "test_dir"
DATA_TRANSFORM_PREPROCESSED_OBJECT_DIR_KEY = "preprocessed_object_dir"
DATA_TRANSFORM_PREPROCESSED_OBJECT_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORM_CLUSTER_MODEL_DIR_KEY = "cluster_model_dir"
DATA_TRANSFORM_CLUSTER_MODEL_NAME_KEY = "cluster_model_name"