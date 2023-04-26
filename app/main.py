import joblib

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, StandardScaler, \
FunctionTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, KNNImputer
from mlxtend.feature_selection import ColumnSelector

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd


class Patient(BaseModel):
	age: int
	gender: str
	hypertension: int
	heart_disease: int
	ever_married: str
	work_type: str
	residence_type: str
	avg_glucose_level: float
	bmi: float
	smoking_status: str

app = FastAPI()

model = joblib.load('stroke_xgb_clf.pkl')

def predict(model, data):
	data = pd.DataFrame({
		'age': [data.age],
		'gender': [data.gender],
		'hypertension': [data.hypertension],
		'heart_disease': [data.heart_disease],
		'ever_married': [data.ever_married],
		'work_type': [data.work_type],
		'Residence_type': [data.residence_type],
		'avg_glucose_level': [data.avg_glucose_level],
		'bmi': [data.bmi],
		'smoking_status': [data.smoking_status]
	})

	label = model.predict(data)[0]
	spam_prob = model.predict_proba(data)

	return {'label': int(label), 'probability': float(spam_prob[0][1].round(3))}

@app.post('/stroke_prediction_query/')
async def stroke_prediction_query(patient: Patient):
	return predict(model, patient)