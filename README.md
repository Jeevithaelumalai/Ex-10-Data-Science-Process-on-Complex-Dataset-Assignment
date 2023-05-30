# Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
## STEP 1 :
Read the given Data

## STEP 2 :
Clean the Data Set using Data Cleaning Process

## STEP 3 :
Apply Feature Generation/Feature Selection Techniques on the data set

## STEP 4 :
Apply EDA /Data visualization techniques to all the features of the data set

# CODE:
```
Developed by:JEEVITHA E
Register No: 212222230054
```
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("/content/StudentsPerformance - StudentsPerformance.csv.csv")
print(data)

data.info()

data.isnull().sum()
```
# data cleaning
```
data['test preparation course']=data['test preparation course'].fillna(data['test preparation course'].mode()[0])
data['math score']=data['math score'].fillna(data['math score']).fillna(data['math score'].mean())
data['writing score']=data['writing score'].fillna(data['writing score']).fillna(data['reading score'].median())

data.isnull().sum()

data.describe()

data.head()
```

# removing outliers
```
Q1=data['math score'].quantile(0.25)
Q3=data['math score'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=data[(data['math score']>=lower) & (data['math score']<=upper)] 
print(df)   #new dataframe.


outliers=data[(data['math score']<lower) | (data['math score']>upper)] 
print(outliers)

df.shape
```

# Feature generation
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
df1=df.copy()
r=['group A','group B','group C','group D','group E']
enc=OrdinalEncoder(categories=[r])
enc.fit_transform(df1[['race/ethnicity']])
df1['neword1']=enc.fit_transform(df1[['race/ethnicity']])
df1 


df2=df1.copy()
le=LabelEncoder()
df2['neword2']=le.fit_transform(df2['race/ethnicity'])
df2

from sklearn.preprocessing import OneHotEncoder
df3=df.copy()
ohe=OneHotEncoder(sparse=False)
enc=pd.DataFrame(ohe.fit_transform(df3[['lunch']]))
df3=pd.concat([df3,enc],axis=1)
df3.head()

!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df4=df.copy()
newdata=be.fit_transform(df4['test preparation course'])
df4=pd.concat([df,newdata],axis=1)
df4.head()
```
# heatmap
```
data.corr()
plt.subplots(figsize=(7,5))
sns.heatmap(data.corr(),annot=True)
```
# Data visualization
# Scatter plot of math score vs. reading score
```
plt.scatter(data['math score'], data['reading score'])
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs. Reading Score')
plt.show()

sns.barplot(x='gender',y='reading score',data=df)

sns.boxplot(x="math score",data=df)
```
# OUTPUT:
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/9666828b-d00c-4878-832e-b0bbfc36158b)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/6d4a1784-5b65-42f5-bb9c-22a73d3064bd)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/25540625-810d-45bb-8e4f-b1bf6b4789d7)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/d0367309-985d-4527-bfd3-27241b7e03bc)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/b39192cc-5fc8-45a8-813a-f8be5dea13e7)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/52aba743-619f-4679-89c9-67690cdb4ab9)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/e819a14a-09ec-4e17-b6ab-831f54b98847)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/02cb2e20-4e33-4adb-b980-4dc646b4a57f)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/c51c7008-bb43-4840-94f3-224193097204)
![image](https://github.com/Jeevithaelumalai/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118708245/a4602c56-0ee0-4957-915b-68e35ded3a4e)
![Uploading image.png…]()
