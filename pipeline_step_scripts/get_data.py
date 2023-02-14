from azureml.core import Run, Workspace, Datastore, Dataset
import pandas as pd
import os
import argparse
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser("Get raw data from a selected datastore and register in AML workspace")
parser.add_argument('--raw_dataset', dest='raw_dataset', required=True)

args, _ = parser.parse_known_args()
raw_dataset = args.raw_dataset

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Connect to default blob datastore
ds = ws.get_default_datastore()

################################# MODIFY #################################

# The intent of this block is to load from a target data source. This
# can be from an AML-linked datastore or a separate data source accessed
# using a different SDK. Any initial formatting operations can be be 
# performed here as well.

# Load Boston Home Prices dataset from Scikit and convert to a pandas data frame
from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

##########################################################################

# Make directory on mounted storage for output dataset
os.makedirs(raw_dataset, exist_ok=True)

# Save modified dataframe
df.to_csv(os.path.join(raw_dataset, 'raw_data.csv'), index=False)