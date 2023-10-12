# Ex.No.1---Data-Preprocessing
## AIM:
```
To perform Data preprocessing in a data set downloaded from Kaggle
```
## REQUIPMENTS REQUIRED:
```
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
```
## RELATED THEORETICAL CONCEPT:
```
Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:
Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :
For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.
```
## ALGORITHM:
```
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train
```
## PROGRAM:
```
DEVELOPED: SRI RANJANI PRIYA.P
REG NO: 212222220049
```
```
import pandas as pd
import numpy as np
df = pd.read_csv("/content/Churn_Modelling.csv")
df.info()
df.isnull().sum()
df.duplicated()
df.describe()
df['Exited'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape
```
## OUTPUT:
## Dataset:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/902aeaf5-6a09-41b3-84f2-3f7bf78e5c89)

## Checking for Null Values:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/b59b2cb4-c04c-4f9e-a0bc-2d2f0421d298)

## Checking for duplicate values:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/b1e124b9-cb2a-42d6-b10e-bf0d35f98e1b)

## Describing Data:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/5346c1cd-6277-4b55-aa74-3a5285c363f9)

## Checking for outliers in Exited Column:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/a1f05b88-6d0c-40d8-9254-971071d98098)

## Normalized Dataset:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/bad819d2-3ef1-4637-b1d5-6433b87ab454)

## Describing Normalized Data:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/52710462-50f6-4549-829d-b830d667b3ab)

## X - Values:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/2f8885ce-ab74-413d-95e7-5226315e8316)

## Y - Value:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/ce93cf51-d52f-41cd-87a5-580fb8c672dc)

## X_train values:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/c7172343-dbb1-4f95-be47-20037889e4b4)

## X_train Size:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/0674bfb8-ba12-43df-8740-82fac40a1bdb)

## X_test values:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/16e7be99-be00-4014-bdf0-c737d9048eb8)

## X_test Size:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/c3caad7c-dc6c-402a-bbe7-6b767e517171)

## X_train shape:

![image](https://github.com/22008008/Ex.No.1---Data-Preprocessing/assets/118343520/0293a7bf-d9bf-4ae2-98b6-8477e842c2e5)

## RESULT
```
Data preprocessing is performed in a data set downloaded from Kaggle.
```
