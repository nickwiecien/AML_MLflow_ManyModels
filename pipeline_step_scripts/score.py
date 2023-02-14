import json
import os
from fnmatch import fnmatch
import mlflow
import mlflow.sklearn

import pandas as pd

def init():
    # Load the model from file or from Azure ML model registry
    global models
    models = {}
    pattern = '*MLModel'
    subdirs = [x[0] for x in os.walk('.') if fnmatch(x[0], pattern)]
    for subdir in subdirs:
        model_name = subdir.split('/')[-1]
        models[model_name] = mlflow.sklearn.load_model(f'{subdir}/model')

    
def run(data):
    # Preprocess the input data
    data = json.loads(data)
    X = pd.DataFrame(json.loads(data['data']))
    model_name = data['model_name']
    # Score the model
    preds = models[model_name].predict(X)
    # Return the results
    return json.dumps(preds.tolist())