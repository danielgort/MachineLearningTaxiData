# MachineLearningTaxiData
Testing out Machine Learning using Tensorflow with some taxi data

## Preparing the data
You can download the raw data from the following site: https://www.kaggle.com/mnavas/taxi-routes-for-mexico-city-and-quito/data
There are also a couple other data sets from different cities that can be used.

Some of the columns were preprocced from uio_clean.csv -> uio_formatted.csv. This included removing some irrelevant columns, adding a calculated column (speed) and converting the date fields to day of week and hour of day.  Please be advised that the date data is this data set is pretty poor, becuase it is not in 24 hr clock and/or does not contain am/pm info.

taxi_data.py takes uio_formatted.csv and splits it into training and test datasets.  It also adds additional calculated feature columns and preps the data based on what field you are estimating.  Also it splits the estimated field into classes if you are using a classifier as the estimator

## Estimator
taxidata_estimator.py allows you to choose from either a classifier or a regressor for your estimator.  You can also pass in a parameter to use a linear model or a neural netwrok or use Tensorflow's combined estimator that uses both a neural network and linear model.
