import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Used_Bikes.csv') 

X = data.drop(['bike_name', 'price', 'city'], axis=1)  
y = data['price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

import pickle

with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)


with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

new_data = {
    'kms_driven': [2000],
    'owner': [1],
    'age': [4],
    'power': [150],
    'brand': [1]
}

new_data_df = pd.DataFrame(new_data)


predictions = loaded_model.predict(new_data_df)

print(predictions[0])
