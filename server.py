from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS from flask_cors
# Load the Random Forest Classifier model
filename = './model.pkl'


with open(filename, 'rb') as f:
    model = pickle.load(f)
print(model)
app = Flask(__name__)
CORS(app)  
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    
    
    # Extract the input data from the request
    age = int(data['age'])
    gender = 1 if data['gender'] == 'male' else 0  # 1=male.....0=female
    chesttype = {'typicalangina': 0, 'atypicalangina': 1, 'nonanginalpain': 2, 'asymptomatic': 3}[data['chesttype']]
    restingbp = int(data['restingbp'])
    cholesterol = int(data['cholesterol'])
    fastingbs = 1 if data['fastingbs'] == 'greater' else 0
    restingecg = {'normal': 0, 'abnormal': 1, 'definite': 2}[data['restingecg']]
    maxhr = int(data['maxhr'])
    exerciseangina = 1 if data['exerciseangina'] == 'yes' else 0
    stdepression = float(data['stdepression'])
    stslope = {'upsloping': 0, 'flat': 1, 'downsloping': 2}[data['stslope']]
    kidney = 1 if data['kidney'] == 'yes' else 0

    # Create the input array
    input_data = np.array([[age, gender, chesttype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, stdepression, stslope, kidney]])
    


    prediction = model.predict(input_data)
    print(prediction)
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
