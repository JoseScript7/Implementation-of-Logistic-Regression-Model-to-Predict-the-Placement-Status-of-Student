# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries & load data using pandas, and preview with df.head().

2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.

3.Encode categorical columns (like gender, education streams) using LabelEncoder.

4.Split features and target:

X = all columns except status

y = status (Placed/Not Placed)

5.Train-test split (80/20) and initialize LogisticRegression.

6.Fit the model and make predictions.

7.Evaluate model with accuracy, confusion matrix, and classification report.

## Program / Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.Ranen Joseph Solomon
RegisterNumber: 212224040269
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/57e17ab9-fce2-499b-bfaa-74bda462ccc5)
```
data1=data.copy()
data1.head()
```
![image](https://github.com/user-attachments/assets/ca0d5494-4025-4321-907f-28b5ad3825df)
```
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/c08227df-ea75-4eea-8426-0508a8799fa0)
```
data1.duplicated().sum()
data1
```
![image](https://github.com/user-attachments/assets/55c6718d-70bf-4b27-a5b8-fcd9d6d0f03a)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![image](https://github.com/user-attachments/assets/84cb0588-186e-4cd6-ada8-9079758eeaa7)
```
x=data1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/5bce68fe-1a54-4db7-a2c1-8d839748fda4)
```
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/c868c299-3632-47aa-9440-a2d432309339)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
```
![image](https://github.com/user-attachments/assets/1829e8d4-9886-4552-bb98-ca9def472037)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
