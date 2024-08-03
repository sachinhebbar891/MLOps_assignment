import joblib
import pandas as pd

def test_prediction():
    scaler = joblib.load('../model/standard_scaler.joblib')
    model = joblib.load('../model/logistic_regression_model.joblib')
    
    new_data = pd.read_csv('../data/test.csv')
    X_test = new_data.drop(['Outcome'], axis = 1)
    y_test = new_data['Outcome'].copy()
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    assert len(predictions) == 2

if __name__ == "__main__":
    test_prediction()
