## Function to predict a customer likelyhood of churning
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


model_file = 'final_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('predict')


@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()
    X  = pd.json_normalize([customer])
    pred = model.predict(X)
    result = (pred[0])
 
    return jsonify(result) 


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

