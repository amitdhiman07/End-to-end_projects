## Diabetes detection 

## Libraries 
import pandas as pd 

import sklearn 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

## Reading file 
data = pd.read_csv(r"C:\Users\Amit\Desktop\Dataset\Diabetes.csv")
df = pd.DataFrame(data)
print("Dataset \n")
print("*"*50)
print(df)

## Checking Data structure and data types 
print("Information of the dataset ", df.info())
print("*"* 20)
print("Checking null values of the dataset \n ", df.isna().sum())
print("*"* 20)
print("Checking the duplicated values of the dataset \n ", df.duplicated().value_counts())

## Checking the correlation between the variables 
print(df.corr())
## based on the correlation between the variables 
## Pregnancies , Glucose , BMI , Age have the moderate positive correlation with the Outcome variable.

## Feature Extraction
X = df.drop(columns=['Pregnancies','BloodPressure' ,'SkinThickness','Insulin','DiabetesPedigreeFunction','Outcome'])
Y = df['Outcome']

print(f"Features are : \n {X}")
print("Target variable is : \n{}".format(Y))

## Splitting data into training and testing sets

X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25,random_state=40)
print(f"Shape of the training sets are : \n X_train {X_train.shape}\n y_train {y_train.shape}")
print(f"\nShape of the testing sets are : \n X_test {X_test.shape}\n y_test {y_test.shape}")


## Model construction

# ## Logistic Regression 
# logi = LogisticRegression()
# logi.fit(X_train,y_train)
# y_pred = logi.predict(X_test)
# #print(y_pred)

# ## Evaluation
# accuracy = accuracy_score(y_test , y_pred)
# print(f"accuracy score for logistic regression is : {accuracy}")

## RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_preds = rfc.predict(X_test)
#print(y_preds)

## Evaluation
print(f"accuracy score for RandomForestClassifier is : {accuracy_score(y_test , y_preds)}")

# #GradientBoostingClassifier
# gbc = GradientBoostingClassifier()
# gbc.fit(X_train,y_train)
# y_pre = gbc.predict(X_test)
# #print(y_pre)

# ## Evaluation
# print(f"accuracy score for GradientBoostingClassifier is : {accuracy_score(y_test , y_pre)}")

# #SVC
# svc = SVC()
# svc.fit(X_train,y_train)
# y_pr = svc.predict(X_test)
# #print(y_pr)

# ## Evaluation
# print(f"accuracy score for SVC is : {accuracy_score(y_test , y_pr)}")

import joblib

# Save the model 
joblib.dump(rfc, 'diabetes_model.pkl')



