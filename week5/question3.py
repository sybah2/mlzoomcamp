import pickle
dv = 'dv.bin'
model = 'model1.bin'


with open(dv, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model, 'rb') as f_in:
    model = pickle.load(f_in)



customer = {"job": "management", "duration": 400, "poutcome": "success"}



X  = dv.transform([customer])
y_pred = model.predict_proba(X)[0,1]

print('input:', customer)
print("churn probability is ", y_pred)