# Step 2. Split Data
# Sample Python script designed to retrieve a pandas dataframe
# containing raw data, then split that into train and test subsets.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
from numpy.random import seed

# Parse input arguments
parser = argparse.ArgumentParser("Split raw data into train/test subsets.")
parser.add_argument('--training_data', dest='training_data', required=True)
parser.add_argument('--testing_data', dest='testing_data', required=True)
parser.add_argument('--testing_size', type=float, required=True)

args, _ = parser.parse_known_args()
training_data = args.training_data
testing_data = args.testing_data
testing_size = args.testing_size

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
raw_datset = current_run.input_datasets['Raw_Data']
raw_df = raw_datset.to_pandas_dataframe()

# Save train data to both train and test dataset locations.
os.makedirs(training_data, exist_ok=True)
os.makedirs(testing_data, exist_ok=True)

################################# MODIFY #################################

# Optionally include data transformation steps here. These may also be
# included in a separate step entirely.
data_cols = [x for x in raw_df.columns if x!='target']

import random
import json

subsets = set()
while len(list(subsets)) <10:
    count = random.randint(6, len(data_cols))
    subset = random.sample(data_cols, count)
    subset.sort()
    subsets.add(json.dumps(subset))
   
subsets = list(subsets)
subsets = [json.loads(x) for x in subsets]

for i in range(0, len(subsets)):
    cols = subsets[i] + ['target']
    filename = f'DATA_{str(i)}.csv'
    sub_df = raw_df[cols]
    train_df, test_df = train_test_split(sub_df, test_size=0.2, random_state=0)
    train_df.to_csv(os.path.join(training_data, filename), index=False)
    test_df.to_csv(os.path.join(testing_data, filename), index=False)


##########################################################################