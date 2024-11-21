## Function to predict a customer likelyhood of churning
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import xgboost as xgb


model_file = 'xboos_model.bin'

#model = load('./xboos_model.bin')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('predict')


@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()
    X  = pd.json_normalize([customer])
    X  = X
    X_try_dict = X.to_dict(orient='records')
    X_try = dv.transform(X_try_dict)
    feature_names = list(dv.get_feature_names_out())
     
    feature_names
    xtest = xgb.DMatrix(X_try, feature_names=feature_names)
    y_pred = model.predict(xtest)[0]
    
    churn = y_pred >= 0.5
 
    result = {
        'Proberbility of churn': round(float(y_pred),3),
        'Churn': bool(churn)
    }
 
    return jsonify(result) 


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)






