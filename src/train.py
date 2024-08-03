import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


# Example training script
def train_model():
    # Load data
    data = pd.read_csv('../data/train.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome'].copy()

    # Initialize and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    # Initialize and fit model
    model = LogisticRegression()
    model.fit(X_train_scaled, y)

    # Save scaler and model
    joblib.dump(scaler, '../model/standard_scaler.joblib')
    joblib.dump(model, '../model/lr_model.joblib')


if __name__ == "__main__":
    train_model()
