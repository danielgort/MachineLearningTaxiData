"""Estimator for taxi dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import math

import taxi_data

nbuckets = 20

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', default=50, type=int, help='batch size')
parser.add_argument('--train_steps', '-ts', default=880, type=int,
                    help='number of training steps')
parser.add_argument('--num_epochs', '-ne', default=5, type=int,
                    help='number of epochs')
parser.add_argument('--learning_rate', '-lr', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('--label_type', '-lt', default='trip_duration', type=str,
                    help='what data you want to predict, options; trip_duration, speed, dist_meters')
parser.add_argument('--model_type', '-mt', default='combined', type=str,
                    help='what type of model to use; options: linear, neural, combined')
parser.add_argument('--hidden_units', '-hu', nargs='+', default=[4096,2048,1024], type=int,
                    help='hidden layers and units in each layer, usage: -hu 4096 2048 1024')
parser.add_argument('--classifier_regressor', '-cr', default='classifier', type=str,
                    help='use a classifier or a regressor for your estimator')

def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Hashing:
    vendor_id = tf.feature_column.categorical_column_with_hash_bucket(
        'vendor_id', hash_bucket_size=1000)


    pickup_dayofweek = tf.feature_column.categorical_column_with_vocabulary_list(
        'pickup_dayofweek', [
            1, 2, 3, 4, 5, 6, 7])

#    pickup_hour = tf.feature_column.categorical_column_with_identity(
#        'pickup_hour', num_buckets = 12)
    pickup_hour = tf.feature_column.categorical_column_with_vocabulary_list(
         'pickup_hour', [
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


    # dropoff_dayofweek = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'dropoff_dayofweek', [
    #        0, 1, 2, 3, 4, 5, 6])

    # dropoff_hour = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'dropoff_hour', [
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


    # Continuous columns
    pickup_longitude = tf.feature_column.numeric_column('pickup_longitude')
    pickup_latitude = tf.feature_column.numeric_column('pickup_latitude')
    dropoff_longitude = tf.feature_column.numeric_column('dropoff_longitude')
    dropoff_latitude = tf.feature_column.numeric_column('dropoff_latitude')
    longitude_diff = tf.feature_column.numeric_column('latitude_diff')
    latitude_diff = tf.feature_column.numeric_column('longitude_diff')
    pseudo_distance = tf.feature_column.numeric_column('pseudo_distance')

    # Bucketize the lats & lons, lat and long have been normalised so distributed between 0 and 1
    lonbuckets = np.linspace(0.0, 1.0, nbuckets).tolist()
    latbuckets = np.linspace(0.0, 1.0, nbuckets).tolist()

    b_pickup_longitude = tf.feature_column.bucketized_column(pickup_longitude, lonbuckets)
    b_pickup_latitude = tf.feature_column.bucketized_column(pickup_latitude, latbuckets)
    b_dropoff_longitude = tf.feature_column.bucketized_column(dropoff_longitude, lonbuckets)
    b_dropoff_latitude = tf.feature_column.bucketized_column(dropoff_latitude, latbuckets)

    # Feature cross
    pickup_loc = tf.feature_column.crossed_column([b_pickup_latitude, b_pickup_longitude], nbuckets * nbuckets)
    dropoff_loc = tf.feature_column.crossed_column([b_dropoff_latitude, b_dropoff_longitude], nbuckets * nbuckets)
    pickup_dropoff_pair = tf.feature_column.crossed_column([pickup_loc, dropoff_loc], nbuckets ** 4 )
    pickup_day_hr =  tf.feature_column.crossed_column([pickup_dayofweek, pickup_hour], 84)

    base_columns = [
        vendor_id, pickup_dayofweek, pickup_hour, 
    ]

    crossed_columns = [pickup_loc, dropoff_loc, pickup_dropoff_pair, 
        pickup_day_hr,
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        tf.feature_column.indicator_column(vendor_id),
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pickup_dropoff_pair, 10),
        tf.feature_column.embedding_column(pickup_day_hr, 10),
        pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, 
        longitude_diff, latitude_diff, pseudo_distance, ]


    return wide_columns, deep_columns

def my_model(features, labels, mode, params):
    """DNN with multiple hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute outputs
    logits = tf.layers.dense(net, 1, activation=None)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': tf.nn.softmax(logits),
            'predictions': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.

    loss = tf.losses.log_loss(
        labels=labels, 
        predictions=tf.reshape(logits,[-1]))

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=logits,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(argv):
    args = parser.parse_args(argv[1:])

    # allow CPU to be used if we cannot use the GPU
    tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Fetch the data
    # (train_x, train_y), (test_x, test_y), minmax_latandlong = taxi_data.load_data(y_name='dist_meters')
    if args.classifier_regressor == 'classifier':
        use_classifier = True
    else:
        use_classifier = False
    
    (train_x, train_y), (test_x, test_y), classes = taxi_data.load_data(y_name=args.label_type, bucketize_labels=use_classifier)



    
    # Feature columns describe how to use the input.
    # wide_feature_columns, deep_feature_columns = build_model_columns(minmax_latandlong)
    wide_feature_columns, deep_feature_columns = build_model_columns()

    # Build a DNN and linear estimator with hidden layers.
    if args.model_type == 'combined' and args.classifier_regressor == 'classifier':
        my_estimator = tf.estimator.DNNLinearCombinedClassifier(
            # wide settings
            linear_feature_columns=wide_feature_columns,
            linear_optimizer=tf.train.FtrlOptimizer(args.learning_rate),
            # deep settings
            dnn_feature_columns=deep_feature_columns,
            dnn_optimizer=tf.train.AdagradOptimizer(args.learning_rate),
            # hidden layers.
            dnn_hidden_units=args.hidden_units,
            model_dir='models/class/combined/' + args.label_type,
            n_classes=6,
            label_vocabulary=classes
            # warm-start settings
            # ,warm_start_from='models/' + args.label_type
            )

    # Build a DNN and linear estimator with hidden layers.
    elif args.model_type == 'combined' and args.classifier_regressor == 'regressor':
        my_estimator = tf.estimator.DNNLinearCombinedRegressor(
            # wide settings
            linear_feature_columns=wide_feature_columns,
            linear_optimizer=tf.train.FtrlOptimizer(args.learning_rate),
            # deep settings
            dnn_feature_columns=deep_feature_columns,
            dnn_optimizer=tf.train.AdagradOptimizer(args.learning_rate),
            # hidden layers.
            dnn_hidden_units=args.hidden_units,
            model_dir='models/regres/combined/' + args.label_type
            # warm-start settings
            # ,warm_start_from='models/' + args.label_type
            )

    # Build a DNN estimator with hidden layers.
    elif args.model_type == 'neural' and args.classifier_regressor == 'classifier':
        my_estimator = tf.estimator.DNNClassifier(
            # deep settings
            feature_columns=deep_feature_columns,
            optimizer=tf.train.AdagradOptimizer(args.learning_rate),
            # hidden layers.
            hidden_units=args.hidden_units,
            model_dir='models/class/neural/' + args.label_type,
            n_classes=6,
            label_vocabulary=classes
            # warm-start settings
            # ,warm_start_from='models/' + args.label_type
            )

    # Build a DNN estimator with hidden layers.
    elif args.model_type == 'neural' and args.classifier_regressor == 'regressor':
        my_estimator = tf.estimator.DNNRegressor(
            # deep settings
            feature_columns=deep_feature_columns,
            optimizer=tf.train.AdagradOptimizer(args.learning_rate),
            # hidden layers.
            hidden_units=args.hidden_units,
            model_dir='models/regres/neural/' + args.label_type
            # warm-start settings
            # ,warm_start_from='models/' + args.label_type
            )

    # Build a linear estimator
    elif args.model_type == 'linear' and args.classifier_regressor == 'classifier':
        my_estimator = tf.estimator.LinearClassifier(
            optimizer=tf.train.FtrlOptimizer(args.learning_rate),
            feature_columns=wide_feature_columns,
            model_dir='models/class/linear/' + args.label_type,
            n_classes=6,
            label_vocabulary=classes
            )
    
    # Build a linear estimator
    elif args.model_type == 'linear' and args.classifier_regressor == 'regressor':
        my_estimator = tf.estimator.LinearClassifier(
            optimizer=tf.train.FtrlOptimizer(args.learning_rate),
            feature_columns=wide_feature_columns,
            model_dir='models/regres/linear/' + args.label_type
            )

        # Build a custom estimator with hidden layers.
        # my_estimator = tf.estimator.Estimator(
    #     model_fn=my_model,
    #     # The directory for storing the model.
    #     model_dir='models/taxi_duration',   
    #     params={
    #         'feature_columns': deep_feature_columns,
    #         # hidden layers
    #         'hidden_units': [1024, 512, 256]
    #     })     


    # Train the Model.
    for i in range(args.num_epochs):
        my_estimator.train(
            input_fn=lambda:taxi_data.train_input_fn(train_x, train_y,
                                                    args.batch_size),
            steps=args.train_steps)

    # Evaluate the model.
    eval_result = my_estimator.evaluate(
        input_fn=lambda:taxi_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('/n')
    print(eval_result)
    print('/n')

    
    # Generate predictions from the model
    
    # adjust to use different snapshots of the test data for predictions 
    i = 30
    

    predictions = my_estimator.predict(
        input_fn=lambda:taxi_data.eval_input_fn(test_x[i:i+9],
                                                labels=None,
                                                batch_size=args.batch_size))

    if use_classifier == True:
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        for pred_dict, expec in zip(predictions, test_y[i:i+9]):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]

            print(template.format(classes[class_id],
                                100 * probability, expec))

    else:
        template = ('\nPrediction is "{}", actual value was "{}"')
        for pred_dict in zip(predictions):
            pred_dict = pred_dict[0]
            prediction = pred_dict['predictions'][0]
            
            # print('\nFeatures:')
            # print(test_x.loc[i])
            print(template.format(round(math.exp(prediction)), round(math.exp(test_y.ix[i,args.label_type]))))
            i = i + 1

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
