---
title: 간단한 classification
date: 2024-04-11 16:25:00 +09:00
categories: [데이터 분석, 분류]
author: yehoon
tags: [Pandas, Scikit-learn, Seaborn]
description: p-value
---


## Breast Cancer Wisconsin (Diagnostic) Data Set

### 0. Settings


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
```


```python
file_name = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'

df = pd.read_csv(
    file_name,
)
```

### 1. EDA

#### 1.1. Preview


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 33 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object 
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
     32  Unnamed: 32              0 non-null      float64
    dtypes: float64(31), int64(1), object(1)
    memory usage: 146.8+ KB
    


```python
df.id.duplicated().sum()
```




    0




```python
# 중복된 id가 없으므로 id column을 제외하고 각 row를 독립적인 데이터로 사용
df = df.drop(['id'], axis=1)
```

##### 1.1.1. 결측치 처리


```python
# 마지막 컬럼 제외
df = df.iloc[:, :-1]
```

##### 1.1.2. Label diagnosis


```python
df.diagnosis.value_counts()
```




    diagnosis
    B    357
    M    212
    Name: count, dtype: int64




```python
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
df.diagnosis.value_counts()
```




    diagnosis
    0    357
    1    212
    Name: count, dtype: int64



##### 1.1.3. Correlations


```python
corr = df.corr()
```


```python
sns.heatmap(corr)
plt.show()
```


    
![png](assets/img/breast_cancer_heatmap.png)
    


### 2. Predict

#### 2.1. Data Selection


```python
corr['diagnosis'].sort_values().tail(10)
```




    concavity_mean          0.696360
    area_mean               0.708984
    radius_mean             0.730029
    area_worst              0.733825
    perimeter_mean          0.742636
    radius_worst            0.776454
    concave points_mean     0.776614
    perimeter_worst         0.782914
    concave points_worst    0.793566
    diagnosis               1.000000
    Name: diagnosis, dtype: float64




```python
idx = corr['diagnosis'].sort_values().tail(10).index
X = df[idx].drop(['diagnosis'], axis=1)
y = df['diagnosis']
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>concavity_mean</th>
      <th>area_mean</th>
      <th>radius_mean</th>
      <th>area_worst</th>
      <th>perimeter_mean</th>
      <th>radius_worst</th>
      <th>concave points_mean</th>
      <th>perimeter_worst</th>
      <th>concave points_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.3001</td>
      <td>1001.0</td>
      <td>17.99</td>
      <td>2019.0</td>
      <td>122.80</td>
      <td>25.38</td>
      <td>0.14710</td>
      <td>184.60</td>
      <td>0.2654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0869</td>
      <td>1326.0</td>
      <td>20.57</td>
      <td>1956.0</td>
      <td>132.90</td>
      <td>24.99</td>
      <td>0.07017</td>
      <td>158.80</td>
      <td>0.1860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1974</td>
      <td>1203.0</td>
      <td>19.69</td>
      <td>1709.0</td>
      <td>130.00</td>
      <td>23.57</td>
      <td>0.12790</td>
      <td>152.50</td>
      <td>0.2430</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.2414</td>
      <td>386.1</td>
      <td>11.42</td>
      <td>567.7</td>
      <td>77.58</td>
      <td>14.91</td>
      <td>0.10520</td>
      <td>98.87</td>
      <td>0.2575</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1980</td>
      <td>1297.0</td>
      <td>20.29</td>
      <td>1575.0</td>
      <td>135.10</td>
      <td>22.54</td>
      <td>0.10430</td>
      <td>152.20</td>
      <td>0.1625</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(13, 10))

for i, col_name in enumerate(X.columns):
    sns.boxplot(y=X[col_name], x=y, ax=axs[i%3][i//3])
    
plt.show()
```


    
![png](assets/img/breast_cancer_boxplot.png)
    


#### 2.2. Machine Learning


```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
pipe = Pipeline([
    ('scale', None),
    ('model', KNeighborsClassifier())
])
```


```python
pipe.get_params()
```




    {'memory': None,
     'steps': [('scale', None), ('model', KNeighborsClassifier())],
     'verbose': False,
     'scale': None,
     'model': KNeighborsClassifier(),
     'model__algorithm': 'auto',
     'model__leaf_size': 30,
     'model__metric': 'minkowski',
     'model__metric_params': None,
     'model__n_jobs': None,
     'model__n_neighbors': 5,
     'model__p': 2,
     'model__weights': 'uniform'}




```python
param_grid = {
    'model__n_neighbors':[5,10],
    'scale': [StandardScaler(), MinMaxScaler()],
#     'model__class_weight': [{0:1, 1:v/2} for v in range(1,5)]
}



mod = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3
)
```


```python
mod.fit(X_train, y_train)
```


```python
pd.DataFrame(mod.cv_results_).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean_fit_time</th>
      <td>0.004918</td>
      <td>0.004554</td>
      <td>0.004715</td>
      <td>0.004649</td>
    </tr>
    <tr>
      <th>std_fit_time</th>
      <td>0.000317</td>
      <td>0.000055</td>
      <td>0.000029</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>mean_score_time</th>
      <td>0.012827</td>
      <td>0.012498</td>
      <td>0.012517</td>
      <td>0.012951</td>
    </tr>
    <tr>
      <th>std_score_time</th>
      <td>0.000298</td>
      <td>0.000277</td>
      <td>0.000072</td>
      <td>0.000264</td>
    </tr>
    <tr>
      <th>param_model__n_neighbors</th>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>param_scale</th>
      <td>StandardScaler()</td>
      <td>MinMaxScaler()</td>
      <td>StandardScaler()</td>
      <td>MinMaxScaler()</td>
    </tr>
    <tr>
      <th>params</th>
      <td>{'model__n_neighbors': 5, 'scale': StandardSca...</td>
      <td>{'model__n_neighbors': 5, 'scale': MinMaxScale...</td>
      <td>{'model__n_neighbors': 10, 'scale': StandardSc...</td>
      <td>{'model__n_neighbors': 10, 'scale': MinMaxScal...</td>
    </tr>
    <tr>
      <th>split0_test_score</th>
      <td>0.943662</td>
      <td>0.929577</td>
      <td>0.93662</td>
      <td>0.929577</td>
    </tr>
    <tr>
      <th>split1_test_score</th>
      <td>0.950704</td>
      <td>0.93662</td>
      <td>0.943662</td>
      <td>0.943662</td>
    </tr>
    <tr>
      <th>split2_test_score</th>
      <td>0.93662</td>
      <td>0.943662</td>
      <td>0.93662</td>
      <td>0.93662</td>
    </tr>
    <tr>
      <th>mean_test_score</th>
      <td>0.943662</td>
      <td>0.93662</td>
      <td>0.938967</td>
      <td>0.93662</td>
    </tr>
    <tr>
      <th>std_test_score</th>
      <td>0.00575</td>
      <td>0.00575</td>
      <td>0.00332</td>
      <td>0.00575</td>
    </tr>
    <tr>
      <th>rank_test_score</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
mod.score(X_test, y_test)
```




    0.951048951048951


