#Simple Linear Regression example

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('USA_Housing.csv') #Read data sourcefile

#df.info() 
#df.describe()
# can be used to get brief explanation of numeric data

#Defining Variables
X=df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]

y=df['Price']

#Importing packages and models for LR
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=101)

from sklearn.linear_model import LinearRegression

# Simple Linear Regression application
lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_,X.columns, columns =["Coeff"])

predictions = lm.predict(X_test)

#Graph plotting difference between expected(y-test) values and predicted(predictions) values
plt.scatter(y_test,predictions)
plt.show()

#Calculating error
from sklearn import metrics

print('mean_absolute_error:')
print(metrics.mean_absolute_error(y_test,predictions))

print('mean_squared_error:')
print(metrics.mean_squared_error(y_test,predictions))

print('root_mean_squared_error:')
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
