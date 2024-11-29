# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision Tree classifier
3. Fit the Data in the model
4. Fit the accuracy score
 
## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Abdul Rasak N
RegisterNumber:  24002896
*/
# Importing required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("datasets/Salary.csv")

# Data overview
print(data.head())
print(data.info())
print(data.isnull().sum())

# Encode categorical 'Position' column
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

# Define features (X) and target (y)
x = data[["Position", "Level"]]
y = data["Salary"]

# Display feature and target data
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train a Decision Tree Regressor model
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Make predictions
y_pred = dt.predict(x_test)
print(f"Predicted values: {y_pred}")

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")

# Test prediction for a new sample
sample_prediction = dt.predict([[5, 6]])
print(f"Prediction for sample [5, 6]: {sample_prediction}")

```

## Output:
```
            Position  Level  Salary
0   Business Analyst      1   45000
1  Junior Consultant      2   50000
2  Senior Consultant      3   60000
3            Manager      4   80000
4    Country Manager      5  110000
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Position  10 non-null     object
 1   Level     10 non-null     int64 
 2   Salary    10 non-null     int64 
dtypes: int64(2), object(1)
memory usage: 372.0+ bytes
None
Position    0
Level       0
Salary      0
dtype: int64
   Position  Level  Salary
0         0      1   45000
1         4      2   50000
2         8      3   60000
3         5      4   80000
4         3      5  110000
   Position  Level
0         0      1
1         4      2
2         8      3
3         5      4
4         3      5
0     45000
1     50000
2     60000
3     80000
4    110000
Name: Salary, dtype: int64
Predicted values: [80000. 45000.]
R2 score: 0.48611111111111116
Prediction for sample [5, 6]: [200000.]
```

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
