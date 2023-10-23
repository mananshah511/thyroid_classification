import os,sys,json
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from thyroid.config.configuration import Configuration
from thyroid.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from thyroid.components.data_ingestion import DataIngestion
from thyroid.components.data_validation import DataValidation

class Pipeline:

    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ThyroidException(sys,e) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ThyroidException(sys,e) from e
    
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.intiate_data_validation()
        except Exception as e:
            raise ThyroidException(sys,e) from e
    
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise ThyroidException(sys,e) from e