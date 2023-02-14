from azureml.core import Run, Workspace, Datastore, Dataset, Model
from azureml.pipeline.core import PipelineRun, StepRun, PortDataReference
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
from mlflow import MlflowClient

def init():
    global current_run, model_lookup_dict
    current_run = Run.get_context()
    model_lookup_dict = {}
    
    child = current_run.parent.find_step_run("Train Models")[0]
    
    data = child.get_output_data('training_output')
    data.download('./tmp')
    path = os.path.join('./tmp', 'azureml', child.id, 'training_output/parallel_run_step.txt')
    new_df = pd.read_csv(path, delimiter=' ', header=None)
    new_df.columns = ['Path', 'ModelName', 'ExperimentName', 'RunName', 'RunID', 'R2', 'MSE', 'ModelPath']
    model_lookup_dict = dict(zip(new_df['ModelName'], [eval(x)[0] for x in new_df['ModelPath']]))
    
def run(input_data):
    # 1.0 Set up output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Loop through each file in the batch
    # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}
        # start_datetime = datetime.datetime.now()

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = file_name + '_MLModel'
        
        experiment_name = file_name + '_ModelTraining'
        run_name = file_name + '_ElasticNetCV'
        
        curr_df = pd.read_csv(csv_file_path)
        X = curr_df.drop(columns=['target'])
        y = curr_df[['target']]
    
        challenger_model = mlflow.sklearn.load_model(model_lookup_dict[model_name])
        
        # Check if champion exists
        client = MlflowClient()
        model_exists = False
        for mv in client.search_model_versions(f"name='{model_name}'"):
            model_exists = True
            
        result = {}
        result['ModelName'] = model_name
        result['ModelUpdated'] = False
            
        if model_exists:
            try:
                champion_model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')

                champion_preds = champion_model.predict(X)
                challenger_preds = challenger_model.predict(X)

                champion_mse = mean_squared_error(y, champion_preds)
                challenger_mse = mean_squared_error(y, challenger_preds)

                if challenger_mse < champion_mse:
                    mlflow.register_model(
                        model_lookup_dict[model_name],
                        model_name
                    )
                    result['ModelUpdated'] = True
            except Exception as e:
                mlflow.register_model(
                    model_lookup_dict[model_name],
                    model_name
                )
                result['ModelUpdated'] = True
                
            
        else:
            mlflow.register_model(
                model_lookup_dict[model_name],
                model_name
            )
            result['ModelUpdated'] = True
        

        result_list.append(result)
        
    return pd.DataFrame(result_list)