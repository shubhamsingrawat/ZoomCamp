# An object of Flask class is our WSGI application.
from flask import Flask, request
import pickle
import numpy as np
import json

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/predict',methods=['POST'])
def serve():
    data = request.get_json(force=True)
    prediction = model.predict([[data['age'],
                                data['hypertension'],
                                data['heart_disease'],
                                data['avg_glucose_level'],
                                data['bmi'],
                                data['Residence_type_Rural'],
                                data['Residence_type_Urban'],
                                data['work_type_Govt_job'],
                                data['work_type_Never_worked'],
                                data['work_type_Private'],
                                data['work_type_Self-employed'],
                                data['work_type_children'],
                                data['smoking_status_Unknown'],
                                data['smoking_status_formerly smoked'],
                                data['smoking_status_never smoked'],
                                data['smoking_status_smokes'],
                                data['ever_married_No'],
                                data['ever_married_Yes'],
                                data['gender_Female'],
                                data['gender_Male'],
                                data['gender_Other']]])
    
    #print(type(prediction[0]))
    print(prediction[0])

    output = prediction[0]
    return json.dumps(output, default=str)

# main driver function
if __name__ == '__main__':
    app.run()