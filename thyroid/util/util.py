import os,sys,dill,yaml
from thyroid.exception import ThyroidException
from thyroid.logger import logging


def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ThyroidException(sys,e) from e