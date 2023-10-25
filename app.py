from flask import render_template,Flask,request
import os,sys,json
from flask_cors import CORS,cross_origin
from thyroid.exception import ThyroidException
from thyroid.logger import logging
from thyroid.pipeline.pipeline import Pipeline
from thyroid.entity.artifact_entity import FinalArtifact
import pandas as pd
import numpy as np
from thyroid.util.util import load_object,preprocessing

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    try:
        data = [str(x) for x in request.form.values()]
        df = pd.read_csv(data[0])
        if not os.path.exists('data.json'):
            return render_template('index.html',output_text = "No model is trained, please start training")
        with open('data.json', 'r') as json_file:
            dict_data = json.loads(json_file.read())

        final_artifact = FinalArtifact(**dict_data)
        logging.info(f"final artifact : {final_artifact}")

        train_df = pd.read_csv(final_artifact.ingested_train_data)
        train_df = train_df.iloc[:,:-1]
        columns = train_df.columns

        len_df = len(df)
        df = pd.concat([df,train_df])
        df = preprocessing(df=df)

        preprocessing_object = load_object(file_path = final_artifact.preprocessing_dir)
        columns = df.columns
        df = df.iloc[:len_df]
        df = preprocessing_object.transform(df)
        df = pd.DataFrame(df,columns=columns)


        cluster_object = load_object(file_path = final_artifact.cluster_model_path)
        cluster_number = cluster_object.predict(df)
        output = []
        for i in range(len(df)):
            model_object = load_object(file_path = final_artifact.export_dir_path[cluster_number[i]])
            x = (np.array(df.iloc[i])).reshape(1, -1)
            output.append((int(model_object.predict(x))))

        for i in range(len(output)):
            if output[i]==0:
                output[i]='negative'
            elif output[i]==1:
                output[i]='compensated_hypothyroid'
            elif output[i]==2:
                output[i]='primary_hypothyroid'
            elif output[i]==3:
                output[i]='secondary_hypothyroid'
        
        return render_template('index.html',output_text = f"Batch output is : {output}")
    except Exception as e:
        raise ThyroidException(sys,e) from e


@app.route('/train',methods=['POST'])
@cross_origin()
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        return render_template('index.html',prediction_text = "Model training completed")
    except Exception as e:
        raise ThyroidException(sys,e) from e

if __name__ == "__main__":
    app.run()