# Midterm Project -- Bryce Whitney
Below is the code and outputs I used to answer each of the questions for the project portion of the midterm. I started by importing the following functions and libraries that I used throughout the questions to load, model, and evaluate the data: 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
```

### Question 1: Import the weatherHistory.csv into a data frame. How many observations do we have?

```python
df = pd.read_csv('weatherHistory.csv')
print("Number of Observations: ", df.shape[0])
df
```
Output: 
```
Number of Observations:  96453
```

### Question 2: In the weatherHistory.csv data how many features are just nominal variables?

Nominal Variables: 
- Summary
- Precip Type
- Daily Summary

### Question 3: If we want to use all the unstandardized observations for 'Temperature (C)' and predict the Humidity the resulting root mean squared error is (just copy the first 4 decimal places):

```python
# Extract the columns
temp = df[['Temperature (C)']]
humidity = df[['Humidity']]

# LinearRegression
model = LinearRegression()
model.fit(temp, humidity)
humidity_pred = model.predict(temp)

# Calculate MSE
rmse = np.sqrt(MSE(humidity, humidity_pred))

print("RMSE using all the unstandardized Temperature (C) observations to predict Humidity: ", rmse)
```
Output: 
```
RMSE using Temperature (C) to predict Humidity:  0.1514437964005473
```

### Question 4: If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state=2020, the Ridge model with alpha=0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):

```python
# Get the data
X = df[['Temperature (C)']].values
y = df[['Humidity']].values

# KFold
kf = KFold(n_splits=20, shuffle=True, random_state=2020)

# Model
model = Ridge(alpha=0.1)
ss = SS()

RMSE = []
for trainidx, testidx in kf.split(X):
    Xtrain = X[trainidx]
    Xtest = X[testidx]
    ytrain = y[trainidx]
    ytest = y[testidx]
    
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    
    RMSE.append(np.sqrt(MSE(ytest, y_pred)))
    
print("Average RMSE on the test sets: ", np.mean(RMSE))
```
Output: 
```
Average RMSE on the test sets:  0.15143825148125584
```

### Question 5: Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 6 decimal places):

```python
# Data
X = df[['Apparent Temperature (C)']].values
y = df[['Humidity']].values

# Model
model = RandomForestRegressor(n_estimators=100, max_depth=50)
kf = KFold(n_splits=10, shuffle=True, random_state=1693)

RMSE = []
for trainidx, testidx in kf.split(X):
    Xtrain = X[trainidx]
    Xtest = X[testidx]
    ytrain = y[trainidx]
    ytest = y[testidx]
    
    model.fit(Xtrain, ytrain.ravel())
    y_pred = model.predict(Xtest)
    
    RMSE.append(np.sqrt(MSE(ytest, y_pred)))
    
print("Average RMSE Random Forest: ", np.mean(RMSE))
```
Output: 
```
Average RMSE Random Forest:  0.14353615526603972
```

### Question 6: Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 5 decimal places):

```python
# Data
X = df[['Apparent Temperature (C)']].values
y = df[['Humidity']].values

# Model
polynomial_features = PolynomialFeatures(degree=6)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=1693)
ss = SS()

RMSE = []
for trainidx, testidx in kf.split(X):
    Xtrain = X_poly[trainidx]
    Xtest = X_poly[testidx]
    ytrain = y[trainidx]
    ytest = y[testidx]
    
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    
    RMSE.append(np.sqrt(MSE(ytest, y_pred)))
    
print("Average RMSE Polynomial Regression: ", np.mean(RMSE))
```
Output: 
```
Average RMSE Polynomial Regression:  0.14346597195873528
```

### Question 7: If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):

```python
# Data
X = df[['Temperature (C)']].values
y = df[['Humidity']].values

# Model
model = Ridge(alpha=0.2)
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
ss = SS()

RMSE = []
for trainidx, testidx in kf.split(X):
    Xtrain = X[trainidx]
    Xtest = X[testidx]
    ytrain = y[trainidx]
    ytest = y[testidx]
    
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    
    RMSE.append(np.sqrt(MSE(ytest, y_pred)))
    
print("Average RMSE Polynomial Regression: ", np.mean(RMSE))
```
Output: 
```
Average RMSE Polynomial Regression:  0.15144461669159875
```

### Question 8:  Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):

```python
# Data
X = df[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']].values
y = df[['Temperature (C)']].values

# Model
ss = SS()

polynomial_features = PolynomialFeatures(degree=6)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=1234)

RMSE = []
i = 1
for trainidx, testidx in kf.split(X):
    Xtrain = X_poly[trainidx]
    Xtest = X_poly[testidx]
    ytrain = y[trainidx]
    ytest = y[testidx]
        
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    
    RMSE.append(np.sqrt(MSE(ytest, y_pred)))
    
print("Average RMSE Polynomial Regression 2: ", np.mean(RMSE))
```
Output: 
```
Average RMSE Polynomial Regression 2:  6.0234742758085495
```

### Question 9: Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):

I ran this code ten times and printed out the scores each time. This is because the random forest has an element of randomness, so running it ten times allows me to get a more accurate representation of what the average may look like. It gives me more confidence results aren't due to randomness. 

```python
# Data
X = df[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']].values
y = df[['Temperature (C)']].values

# Model
model = RandomForestRegressor(n_estimators=100, max_depth=50)
kf = KFold(n_splits=10, shuffle=True, random_state=1234)

for i in range(0,10):
    RMSE = []

    for trainidx, testidx in kf.split(X):
        Xtrain = X[trainidx]
        Xtest = X[testidx]
        ytrain = y[trainidx]
        ytest = y[testidx]

        Xtrain = ss.fit_transform(Xtrain)
        Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain.ravel())
        y_pred = model.predict(Xtest)

        RMSE.append(np.sqrt(MSE(ytest, y_pred)))

    print("Average RMSE Random Forest 2: ", np.mean(RMSE))
```
Output: 
```
Average RMSE Random Forest 2:  5.8312256918131515
Average RMSE Random Forest 2:  5.833871649039011
Average RMSE Random Forest 2:  5.833459108016011
Average RMSE Random Forest 2:  5.830842714243756
Average RMSE Random Forest 2:  5.832954158675077
Average RMSE Random Forest 2:  5.832068049454925
Average RMSE Random Forest 2:  5.829805967689619
Average RMSE Random Forest 2:  5.833821255285307
Average RMSE Random Forest 2:  5.8349054365606765
Average RMSE Random Forest 2:  5.8314367286780655
```

### Question 10: If we visualize a scatter plot for Temperature (on the horizontal axis) vs Humidity (on the vertical axis) the overall trend seems to be...
Below is the code I used to visualize the data. 
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

X = df[['Temperature (C)']].values
y = df[['Humidity']].values

plt.subplots()
plt.scatter(X, y)
plt.xlabel("Temp")
plt.ylabel("Humidity")
plt.show()
```
The plot highlighted there is a negative relationship bewteen temperature and humidity.
