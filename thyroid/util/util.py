import os,sys,dill,yaml
from thyroid.exception import ThyroidException
from thyroid.logger import logging
import pandas as pd
from thyroid.constant import DROP_COLUMN_LIST
import numpy as np


def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ThyroidException(sys,e) from e
    
def write_yaml_file(file_path:str, data:dict=None):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        
        with open(file_path,"w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise ThyroidException(sys,e) from e  
    
def load_object(file_path:str):
    try:
        with open(file_path,"rb") as object_file:
            return dill.load(object_file)
    except Exception as e:
        raise ThyroidException(sys,e) from e
    
def preprocessing(df:pd.DataFrame):
    try:
        #preprocessing on test data

        #drop columns
        df.drop(DROP_COLUMN_LIST,inplace=True,axis=1)
        df = df.replace('?', np.NaN)

        #columns mapping
        df['sex'] = df['sex'].map({'F':0,'M':1})
        for columns in df.columns:
            if len(df[columns].unique())==2:
                df[columns] = df[columns].map({'f':0,'t':1})

        #onehot encoding
        df = pd.get_dummies(df,columns=['referral_source'],drop_first=True)

        return df
    except Exception as e:
        raise ThyroidException(sys,e) from e