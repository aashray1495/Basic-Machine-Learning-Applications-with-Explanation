#Logistic Regression example

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')

train.describe()

#Cleaning Data: using heatmap to view NaN values
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.plt.show()

#Functiion to compensate for NULL values
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[0]

    if pd.isnull(Age):

       if Pclass==1:
         return 37
       elif Pclass==2:
         return 29
       else: 
         return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)

#Revisiting heatmap to see semi-cleaned data
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.plt.show()

#Dropping Cabin column due to too many missing values
train.drop('Cabin', axis=1, inplace=True)
#Dropping Null row in Embarked column
train.dropna(inplace=True)

#Converting categorical values to numeric values 
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train=pd.concat([train,sex,embark], axis=1)
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True) #Dropping columns that won't be needed

#Applying Regression

X=train.drop('Survived', axis=1)
y=train['Survived']  

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
