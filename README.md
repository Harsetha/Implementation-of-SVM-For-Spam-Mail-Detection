# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.
2.Import the necessary packages.
3.Read the given csv file and display the few contents of the data.
4.Assign the features for x and y respectively.
5.Split the x and y sets into train and test sets.
6.Convert the Alphabetical data to numeric using CountVectorizer.
7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8.Find the accuracy of the model.
9.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARSETHA J
RegisterNumber:  212223220032
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![444872269-391d20dd-35eb-496f-ae9f-955fb06a4a2c](https://github.com/user-attachments/assets/da35b4a2-eb74-4927-b4e1-a4c01da3ab4d)

![444872280-f9986560-6b08-4dd1-9f8a-afb9bdea3acc](https://github.com/user-attachments/assets/1e006075-c61a-4339-81a7-647e9c5c691a)

![444872290-c867d3ae-602b-4ed6-9611-6af804adbf48](https://github.com/user-attachments/assets/8f59f58f-883c-4127-bc7d-3ef30fb33c44)

![444872510-1faf129a-a34e-4972-9063-53805dbb1ff6](https://github.com/user-attachments/assets/1ec46bc5-541f-41a1-930b-1c1ad74431f4)

![444872520-ffaa37bf-f63d-44db-8771-6f8d4e6b4827](https://github.com/user-attachments/assets/3d9a2473-ede1-4760-9f47-af86223f520b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
