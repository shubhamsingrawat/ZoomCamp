import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer


@load
def sample_data():
    data = pd.read_csv('healthcare-dataset.csv')
    return data


@transform
def features_transform(data):
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
	
	return df


@train
def model_build(df):
	## Split the data
	X = df.drop('stroke',axis=1)
	y = df['stroke']
	X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,random_state=42) 

	## Train the model
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)

	## Predict the result
	y_pred = logreg.predict(X_valid)

	## Performance metrics
	accuracy = accuracy_score(y_valid, y_pred)
	print(accuracy)

    return logreg


@save
def save_model(logreg):
    with open('models/logreg.bin', 'wb') as f_out:
    dump(logreg, f_out)