import pandas as pd


def modify_data():
    data = pd.read_csv('combined_train_data.csv')
    data.drop(data.tail(5).index,
            inplace = True)
    data.to_csv('combined_train_data.csv', index=False)


modify_data()
