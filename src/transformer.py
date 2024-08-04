import pandas as pd
from sklearn.preprocessing import StandardScaler

def modify_data():
    data = pd.read_csv('combined_train_data.csv')
    data.drop(data.tail(5).index,
            inplace = True)
    data.to_csv('combined_train_data.csv', index=False)

modify_data()

# def train_model():
#     # Load data
#     data = pd.read_csv('data/train.csv')
#     X = data.drop('Outcome', axis=1)
#     y = data['Outcome'].copy()

#     # Initialize and fit scaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X)

#     # Combine scaled features and target into a single DataFrame
#     combined_df = pd.DataFrame(X_train_scaled, columns=X.columns)
#     combined_df['Outcome'] = y.reset_index(drop=True)

#     # Save the combined DataFrame to a CSV file
#     combined_df.to_csv('combined_train_data.csv', index=False)

# train_model()