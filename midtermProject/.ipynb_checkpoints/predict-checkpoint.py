## Function to predict a customer likelyhood of churning
def predict(df, model, dv):
    X  = df
    X_try_dict = X.to_dict(orient='records')
    X_try = dv.transform(X_try_dict)
    feature_names = list(dv.get_feature_names_out())
     
    feature_names
    xtest = xgb.DMatrix(X_try, feature_names=feature_names)
    return model.predict(xtest)[0]



