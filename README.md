# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sahana S
RegisterNumber: 25013621
*/
```
~~~
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Size_sqft': [800, 900, 1200, 1500, 1800, 2000, 2300, 2600],
    'Bedrooms': [1, 2, 2, 3, 3, 4, 4, 5],
    'Bathrooms': [1, 1, 2, 2, 3, 3, 4, 4],
    'Price': [500000, 550000, 700000, 850000, 1000000, 1150000, 1300000, 1500000],
    'Occupants': [2, 3, 3, 4, 5, 6, 6, 7]
}

df = pd.DataFrame(data)
df

X = df[['Size_sqft', 'Bedrooms', 'Bathrooms']]
y = df[['Price', 'Occupants']]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(
    max_iter=1000,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)

model = MultiOutputRegressor(sgd)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Predicted values:\n", y_pred)
print("\nActual values:\n", y_test.values)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# New house data: [Size_sqft, Bedrooms, Bathrooms]
new_house = np.array([[1600, 3, 2]])

new_house_scaled = scaler.transform(new_house)
prediction = model.predict(new_house_scaled)

print("Predicted House Price:", prediction[0][0])
print("Predicted Number of Occupants:", int(round(prediction[0][1])))
~~~

## Output:
<img width="429" height="171" alt="image" src="https://github.com/user-attachments/assets/78255a5f-89ab-4662-a2bd-e9373fd29ae4" />
<img width="401" height="64" alt="image" src="https://github.com/user-attachments/assets/c4b9b0fa-d7e3-418e-841a-ce70a99489bb" />
<img width="429" height="57" alt="image" src="https://github.com/user-attachments/assets/440ca5ec-a598-4c9f-8cdd-9588a6325dfb" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
