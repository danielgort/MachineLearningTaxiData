# Copyright 2018 Daniel Gort. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import math

# train_path = r'''C:\Code\Sandbox\Tensorflow\TaxiDataPredictor\RawData\uio_training_data.csv'''
# test_path = r'''C:\Code\Sandbox\Tensorflow\TaxiDataPredictor\RawData\uio_test_data.csv'''

alldata_path = './RawData/uio_formatted.csv'

CSV_COLUMN_NAMES = pd.read_csv(alldata_path, nrows=1).columns

#['id', 'vendor_id', 'pickup_datetime', 'pickup_dayofweek', 'pickup_hour',
#       'dropoff_datetime', 'dropoff_dayofweek', 'dropoff_hour',
#       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
#       'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration',
#       'dist_meters', 'wait_sec', 'speed']

DURATION_COLUMN_NAMES = ['vendor_id', 
    'pickup_dayofweek', 'pickup_hour',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'wait_sec',
    'trip_duration']

DISTANCE_COLUMN_NAMES = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'wait_sec',
    'dist_meters']

# WAIT_COLUMN_NAMES = ['vendor_id', 'pickup_dayofweek', 'pickup_hour',
#     'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
#     'wait_sec']

SPEED_NAMES = ['vendor_id', 'pickup_dayofweek', 'pickup_hour',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'wait_sec',
    'speed']

# Create calculated features
def add_calculated(features,y_name,bucketize_labels):
    # this is how you can do feature engineering in TensorFlow
    lat_start = features['pickup_latitude']
    lat_end = features['dropoff_latitude']
    lon_start = features['pickup_longitude']
    lon_end = features['dropoff_longitude']
    latdiff = (lat_start - lat_end)
    londiff = (lon_start - lon_end)
    
    # set features for distance with sign that indicates direction
    features['latitude_diff'] = latdiff
    features['longitude_diff'] = londiff

    pseudo_distance = (latdiff * latdiff + londiff * londiff).apply(np.sqrt)
    features['pseudo_distance'] = pseudo_distance

    classes = []

    result = features.copy()
    # normailze all the feature columns that are floating point
    for feature_name in features.columns:
        if (features[feature_name].dtype == float or feature_name == 'wait_sec')  and feature_name != y_name:
            max_value = features[feature_name].max()
            min_value = features[feature_name].min()
            result[feature_name] = (features[feature_name] - min_value) / (max_value - min_value)
        #use the log value for output if we are not using buckets
        elif feature_name == y_name:
            if bucketize_labels == False:
                result[feature_name] = np.log(features[feature_name])
            elif y_name == 'trip_duration': #or y_name == 'wait_sec':
                classes = ['<5 mins', '5-10 mins', '10-15 mins', 
                                '15-20 mins', '20-30 mins', '>30 mins']
                result[feature_name] = pd.cut(
                    features[feature_name], 
                    labels=classes, 
                    bins=[0, 300, 600, 900, 1200, 1800, 999999])
            elif y_name == 'dist_meters':
                classes = ['<2 km', '2-3 kms', '3-4 kms', 
                                '4-6 kms', '6-10 kms', '>10 kms']
                result[feature_name] = pd.cut(
                    features[feature_name], 
                    labels=classes, 
                    bins=[0, 2000, 3000, 4000, 6000, 10000, 9999999])
            elif y_name == 'speed':
                classes = ['<15 kms/hr','15-25 kms/hr', '25-35 kms/hr', '35-45 kms/hr', '45-55 kms/hr', '>60 kms/hr']
                result[feature_name] = pd.cut(
                    features[feature_name],
                    labels=classes, 
                    bins=[0, 15, 25, 35, 45, 55, 9999])

    return result, classes

def load_data(y_name='trip_duration', bucketize_labels=False):
    """Returns the taxi dataset as (train_x, train_y), (test_x, test_y)."""

    if y_name=='trip_duration':
        columnsToDrop = ['dropoff_dayofweek', 'dropoff_hour', 
            'dist_meters','speed']
    elif y_name=='dist_meters':
        columnsToDrop = [#'vendor_id', 'pickup_dayofweek', 'pickup_hour',
            'dropoff_dayofweek', 'dropoff_hour', 
            'trip_duration','speed']
    elif y_name=='speed':
        columnsToDrop = ['dropoff_dayofweek', 'dropoff_hour', 
            'trip_duration','dist_meters']


    df = pd.read_csv(alldata_path, names=CSV_COLUMN_NAMES, header=0)
    df.drop(columnsToDrop, inplace=True, axis=1)
    
    df, classes = add_calculated(df,y_name,bucketize_labels)

    # split the dataframe randomnly into 80% for training and the rest for test data
    train=df.sample(frac=0.8,random_state=200)
    test=df.drop(train.index)

    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    # split into features and label
    train_x, train_y = train, train.pop(y_name)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y), classes
    
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(10*batch_size).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



