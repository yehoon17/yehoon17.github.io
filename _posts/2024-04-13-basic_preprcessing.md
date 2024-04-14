---
title: 기본적인 데이터 전처리
date: 2024-04-13 17:25:00 +09:00
categories: [데이터 분석, 전처리]
author: yehoon
tags: [Pandas]
description: 기본적인 데이터 전처리
---

## 1. Data Cleaning:
   - **중복 제거**:
     ```python
     df.drop_duplicates(inplace=True)
     ```

   - **결측치 처리**:
     ```python
     # 결측치를 포함한 행 제거
     df.dropna(inplace=True)

     # 결측치 채우기(특정 값으로 채우거나 인접한 값으로 채우거나)
     df.fillna(
        value=0, 
        # method='ffill' 
     ) 

     # 선형 보간법
     df.interpolation()

     # 통계값으로 채우기
     from sklearn.impute import SimpleImputer
     imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
     df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
     ```

## 2. 스케일링:
   - **Min-Max Scaling**:
     ```python
     from sklearn.preprocessing import MinMaxScaler
     scaler = MinMaxScaler()
     df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
     ```

   - **Standard Scaling**:
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
     ```

## 3. 이상치 확인:
   - **Box Plot**:
     ```python
     import seaborn as sns
     sns.boxplot(data=df)
     ```

   - **Z-Score Method**:
    만약 데이터의 Z-점수가 임계값보다 크다면, 그 데이터는 평균으로부터 표준 편차 단위로 멀리 떨어져 있다는 것을 의미한다.    
    이 방법은 데이터가 정규 분포를 따른다고 가정하고 있으며, 정규 분포를 따르지 않는 데이터셋에 대해서는 잘 작동하지 않을 수 있다고 한다.
     ```python
     from scipy import stats
     z_scores = stats.zscore(df)
     abs_z_scores = np.abs(z_scores)
     outliers = (abs_z_scores > 3).all(axis=1)
     df_no_outliers = df[~outliers]
     ```
