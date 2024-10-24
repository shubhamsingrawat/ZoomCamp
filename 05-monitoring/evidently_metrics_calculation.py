
import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import QuantileTransformer

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

with open('models/logreg.bin', 'rb') as f_in:
	model = joblib.load(f_in)

data = pd.read_csv('healthcare-dataset.csv')

imputer = SimpleImputer(strategy = 'mean')
data['bmi']=imputer.fit_transform(data[['bmi']])
encoded_data= data.copy()

features_to_scale=['age','bmi']
scaler = MinMaxScaler()
encoded_data[features_to_scale]=scaler.fit_transform(encoded_data[features_to_scale])

scaler = QuantileTransformer(output_distribution='uniform')

encoded_data['avg_glucose_level'] = scaler.fit_transform(encoded_data[['avg_glucose_level']])
df = encoded_data.copy()
columns_to_encode = ['Residence_type', 'work_type', 'smoking_status','ever_married','gender']

for column in columns_to_encode:
    encoded_column = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded_column], axis=1)
    df = df.drop(columns=[column],axis=1)

df = df.astype(int)
df.drop('id',axis=1,inplace=True)

## Split the data
X = df.drop('stroke',axis=1)
y = df['stroke']
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,random_state=42) 

## Train the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
X_train['prediction'] = y_train
## Predict the result
y_pred = logreg.predict(X_valid)
X_valid['prediction'] = y_pred
begin = datetime.datetime(2024, 7, 2, 0, 0)


column_mapping = ColumnMapping(
    target=None,
    prediction='prediction'
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i):

	report.run(reference_data=X_train, current_data=X_valid, column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
