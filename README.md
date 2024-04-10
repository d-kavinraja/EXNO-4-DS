# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.
# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.
# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).
# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## KAVINRAJA D [212222240047]

**Feature Scaling**
```py
import pandas as pd
from scipy import stats
import numpy as np
```
```py
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/376dd65a-8a46-4e5a-aa06-89db3bf431f6)
```py
df.head()
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/33268cf2-cc37-41c2-bad9-cbd2601a933c)
```py
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/ff3d3a86-1323-4af3-995f-dd5681da7c86)
```py
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/d1399553-bb11-495a-99e5-622ff96c20b1)
```py
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/8172543a-3361-49da-97be-84430ddb9f61)
```py
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/68ce3578-0f70-48f2-b9ea-1a678840c015)
```py
df=pd.read_csv("/content/bmi.csv")
```
```py
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/1aedf04d-036c-4544-a2d5-bb05ed2663cd)
```py
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-04-08 131902](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/f6d88d95-f69b-4f18-b82a-f9ba84200203)


**Feature Selection**
```py
import pandas as pd
import numpy as np
import seaborn as sns
```
```py
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```py
data=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
data
```
![Screenshot 2024-04-08 132056](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/76a6f995-f13b-4c95-ae8c-528e72850c3b)
```py
data.isnull().sum()
```
![Screenshot 2024-04-08 132139](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/9f1b357d-a955-4d6b-8f37-d9c2900fdf03)
```py
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-04-08 132213](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/588e9033-2d09-4caf-a854-8cd0c5620418)
```py
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-04-08 132301](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/f904cb0a-179c-4b18-9758-e4871b88ff35)
```py
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than  or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-04-08 132337](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/5697629f-598e-49da-98f5-f6f9736d4ef3)
```py
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-04-08 132411](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/9ace2ee3-17e9-4591-b86d-d995ac6b7ff5)
```py
data2
```
![Screenshot 2024-04-08 132437](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/954ae119-4c01-4e3d-9430-f8be6cede5eb)
```py
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![Screenshot 2024-04-08 132540](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/fb035e04-7b58-4fa2-a856-5f1258d98d70)
```py
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-04-08 132627](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/f670491b-3a15-4940-9442-af6132c37327)
```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-04-08 132703](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/8861beaf-4ff8-46d5-8f32-f219cb7c0729)
```py
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-04-08 132737](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/e4ee6b46-e4d5-41a3-a961-5e3988f5ee98)
```py
x=new_data[features].values
print(x)
```
![Screenshot 2024-04-08 132839](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/b794a8cb-8325-45bf-958d-1c2305f5e20a)
```py
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_Classifier.predict(test_x)
```
```py
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![Screenshot 2024-04-08 133507](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/3906c6b9-5480-498d-b139-e7db823e3b05)
```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-04-08 133558](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/fbb86aca-9120-40ee-b59e-3f600de10e3b)
```py
print('Misclassified samples: %d'% (test_y != prediction).sum())
```
![Screenshot 2024-04-08 133630](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/7f519b52-847b-4b3c-bd14-783788d19900)
```py
data.shape
```
![Screenshot 2024-04-08 133707](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/7f96213c-becc-455b-9be3-db54bed1b22d)
```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-04-08 135252](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/0d2359e5-5b37-4091-9203-6e65df36d92f)

```py
tips.time.unique()
```
![Screenshot 2024-04-08 135612](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/9836a9f3-fe9e-4ec3-920d-2a3530762734)
```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-04-08 135823](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/10c74211-c4bb-4a26-b5cf-784f7204cc59)
```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-04-08 140616](https://github.com/arun1111j/EXNO-4-DS/assets/128461833/469760f6-4232-4c33-af9d-b8a8dcfd7d17)







# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
