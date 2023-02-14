from azureml.core import Run, Workspace, Datastore, Dataset, Model
import pandas as pd
import os
import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def init():
    global current_run
    current_run = Run.get_context()
    
def run(input_data):
    # 1.0 Set up output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Loop through each file in the batch
    # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = file_name + '_MLModel'
        
        experiment_name = file_name + '_ModelTraining'
        run_name = file_name + '_ElasticNetCV'
        
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:

            curr_df = pd.read_csv(csv_file_path)
            X = curr_df.drop(columns=['target'])
            y = curr_df[['target']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # create a pipeline object
            pipeline = Pipeline([
                ('scaler', MinMaxScaler()), # apply a minmax scaler
                ('model', ElasticNetCV(cv=3, random_state=0)) # train a ElasticNetCV model
            ])

            # fit the pipeline to the data
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric('mse', mse)
            mlflow.log_metric('r2', r2)

            mlflow.sklearn.log_model(pipeline, "model", 
                                     extra_pip_requirements=[
                                         "azureml-defaults==1.47.0",
                                         "opencensus==0.11.0",
                                         "pandas==1.4.2",
                                         "scikit-learn==0.22.1",
                                         "numpy==1.21.6"
                                     ])
            run_id = run.info.run_id

            result = {}
            result['source_file'] = csv_file_path
            result['model_name'] = model_name
            result['experiment_name'] = experiment_name
            result['run_name'] = run_name
            result['run_id'] = run.info.run_id
            result['r2'] = r2
            result['mse'] = mse
            result['mlflow_model_path'] = f"runs:/{run_id}/model", model_name
            result_list.append(result)
        
    return pd.DataFrame(result_list)