import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


# Example prediction script
def predict(new_data):
    scaler = joblib.load('../model/standard_scaler.joblib')
    model = joblib.load('../model/lr_model.joblib')

    X_test = new_data.drop(['Outcome'],axis=1)
    y_test = new_data['Outcome'].copy()
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy


if __name__ == "__main__":
    # Example usage
    new_data = pd.read_csv('../data/test.csv')
    predictions, accuracy = predict(new_data)
    print(predictions)
    print(accuracy)
