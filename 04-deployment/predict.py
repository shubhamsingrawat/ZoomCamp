####### Old version of code ########
'''
# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import logging

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
    logging.info('123')
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([45,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
'''