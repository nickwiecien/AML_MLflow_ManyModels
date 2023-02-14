from azureml.core import Run, Workspace, Datastore, Dataset, Model, Environment
import pandas as pd
import os
import argparse
import numpy as np
import shutil

# Parse input arguments
parser = argparse.ArgumentParser("Get raw data from a selected datastore and register in AML workspace")
parser.add_argument('--deployment_name', type=str, required=True)

args, _ = parser.parse_known_args()
deployment_name = args.deployment_name

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Connect to default blob datastore
ds = ws.get_default_datastore()

os.mkdir('./models')

training_files = os.listdir('./training-data')
print(training_files)

for file in training_files:
    file_name = os.path.basename(file)[:-4]
    model_name = file_name + '_MLModel'
    model = Model(ws, model_name)
    model.download(target_dir=f'./models/{model_name}')
    
env = Environment.from_conda_specification(name = 'MLflow_Custom_Env', file_path = f'./models/{model_name}/model/conda.yaml')
env.register(ws)

import datetime
# Create a current timestamp
ts = datetime.datetime.now()
# Format it as a string with year, month, day, hour, minute, second and microsecond
deployment_version = ts.strftime('%Y%m%d%H%M%S%f')
    
from azureml.core import Environment
from azureml.core.model import InferenceConfig

source = 'score.py'
destination = './models'
shutil.copy(source, destination)

inference_config = InferenceConfig(
    environment=env,
    source_directory="./models",
    entry_script="score.py",
)

package = Model.package(ws, [], inference_config, image_name=deployment_name, image_label=deployment_version)
package.wait_for_creation(show_output=True)
location = package.location
    