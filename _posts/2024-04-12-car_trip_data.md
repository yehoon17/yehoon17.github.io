---
title: Car trips data log 확인
date: 2024-04-12 14:25:00 +09:00
categories: [데이터 분석, EDA]
author: yehoon
tags: [Pandas, Seaborn]
description: 케글의 Car trips data log 확인
---

데이터: <https://www.kaggle.com/datasets/vitorrf/cartripsdatamining>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
```


```python
df_list = []

for i in range(10, 15):
    file_name = f'/kaggle/input/cartripsdatamining/Processed Data/fileID{i}_ProcessedTripData.csv'

    df = pd.read_csv(
        file_name,
        header= None,
    )
    
    df.columns=['Time','Vehicle Speed','SHIFT','Engine Load','Total Acceleration',
            'Engine RPM','Pitch','Lateral Acceleration','Passenger Count',
            'Car Load','AC Status','Window Opening','Radio Volume','Rain Intensity',
            'Visibility','Driver Wellbeing','Driver Rush']
    
    df_list.append(df)
```


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
      <th>Time</th>
      <th>Vehicle Speed</th>
      <th>SHIFT</th>
      <th>Engine Load</th>
      <th>Total Acceleration</th>
      <th>Engine RPM</th>
      <th>Pitch</th>
      <th>Lateral Acceleration</th>
      <th>Passenger Count</th>
      <th>Car Load</th>
      <th>AC Status</th>
      <th>Window Opening</th>
      <th>Radio Volume</th>
      <th>Rain Intensity</th>
      <th>Visibility</th>
      <th>Driver Wellbeing</th>
      <th>Driver Rush</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.019</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.050212</td>
      <td>-0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.026</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.051910</td>
      <td>-0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.037</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.053624</td>
      <td>-0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.048</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.055352</td>
      <td>-0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.056</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.057097</td>
      <td>-0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 118551 entries, 0 to 118550
    Data columns (total 17 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   Time                  118551 non-null  float64
     1   Vehicle Speed         118551 non-null  float64
     2   SHIFT                 118551 non-null  int64  
     3   Engine Load           118551 non-null  float64
     4   Total Acceleration    118551 non-null  float64
     5   Engine RPM            118551 non-null  float64
     6   Pitch                 118551 non-null  float64
     7   Lateral Acceleration  118551 non-null  float64
     8   Passenger Count       118551 non-null  int64  
     9   Car Load              118551 non-null  int64  
     10  AC Status             118551 non-null  int64  
     11  Window Opening        118551 non-null  int64  
     12  Radio Volume          118551 non-null  int64  
     13  Rain Intensity        118551 non-null  int64  
     14  Visibility            118551 non-null  int64  
     15  Driver Wellbeing      118551 non-null  int64  
     16  Driver Rush           118551 non-null  int64  
    dtypes: float64(7), int64(10)
    memory usage: 15.4 MB
    


```python
cols = ['Vehicle Speed','SHIFT','Engine Load','Total Acceleration',
            'Engine RPM','Pitch','Lateral Acceleration']

fig, axs = plt.subplots(
    nrows=len(cols), 
    figsize=(8, 2*len(cols)), 
    sharex='col',
    sharey='row',
)

for ax, col in zip(axs, cols):
    sns.lineplot(df[:2000], x='Time', y=col, ax=ax)
    
plt.tight_layout()
plt.show()
```


```python
from scipy.integrate import cumtrapz
```


```python
df['Distance'] = cumtrapz(df['Vehicle Speed'], df['Time'], initial=0)
```


```python
cols = ['Vehicle Speed','Distance']

fig, axs = plt.subplots(
    nrows=len(cols), 
    figsize=(8, 2*len(cols)), 
    sharex='col',
    sharey='row',
)

for ax, col in zip(axs, cols):
    sns.lineplot(df[:4000], x='Time', y=col, ax=ax)
    
plt.tight_layout()
plt.show()
```


    
![png](/assets/img/graph/time_to_dist_and_speed.png)
    



```python
df['Abs Acceleration'] = df['Total Acceleration'].abs()

sns.lineplot(df[:4000], x='Distance', y='Abs Acceleration')
plt.fill_between(df[:4000]['Distance'], df[:4000]['Abs Acceleration'], color='skyblue', alpha=0.3)
plt.show()
```


    
![png](/assets/img/graph/time_to_acc.png)
    



```python
from scipy.integrate import trapz

work = trapz(df['Abs Acceleration'], df['Distance'])
work
```




    5260.302733112259




```python
df['Work'] = cumtrapz(df['Abs Acceleration'], df['Distance'], initial=0)
```


```python
def get_dist_col(df):
    df['Distance'] = cumtrapz(df['Vehicle Speed'], df['Time'], initial=0)
    return df

def get_work_col(df, to_abs=True):
    if to_abs:
        df['Work'] = cumtrapz(df['Total Acceleration'].abs(), df['Distance'], initial=0)
    else:
        df['Work'] = cumtrapz(df['Total Acceleration'], df['Distance'], initial=0)
    return df
```


```python
df_list = [get_dist_col(df) for df in df_list]
df_list = [get_work_col(df) for df in df_list]
```


```python
df = df_list[2]
```


```python
cols = ['Vehicle Speed','SHIFT','Engine Load','Total Acceleration',
            'Engine RPM','Pitch','Lateral Acceleration']

fig, axs = plt.subplots(
    nrows=len(cols), 
    figsize=(8, 2*len(cols)), 
    sharex='col',
    sharey='row',
)

for ax, col in zip(axs, cols):
    sns.lineplot(df[:2000], x='Time', y=col, ax=ax)
    
plt.tight_layout()
plt.show()
```


    
![png](/assets/img/graph/time_to_all.png)
    

