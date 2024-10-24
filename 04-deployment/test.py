import requests

details = {'age':45,
          'hypertension':0,
          'heart_disease':1,
          'avg_glucose_level':0,
          'bmi':0,
          'Residence_type_Rural':0,
          'Residence_type_Urban':1,
          'work_type_Govt_job':0,
          'work_type_Never_worked':0,
          'work_type_Private':1,
          'work_type_Self-employed':0,
          'work_type_children':0,
          'smoking_status_Unknown':0,
          'smoking_status_formerly smoked':0,
          'smoking_status_never smoked':0,
          'smoking_status_smokes':1,
          'ever_married_No':0,
          'ever_married_Yes':1,
          'gender_Female':0,
          'gender_Male':1,
          'gender_Other':0,}

url = 'http://localhost:5000/predict'


r = requests.post(url,json=details)
print(r.text)